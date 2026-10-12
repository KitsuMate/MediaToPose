"""Convert canonical detections into Blender pose and shape-key channels."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from . import (
    animation_cleanup,
    camera_motion_correction,
    detection_cache,
    grounded_cleanup,
    mapping,
    motion_filter,
    rig_planning,
)
from .constants import (
    BODY_JOINT_PAIRS,
    HAND_JOINT_PAIRS,
    PROFILE_ARP,
    PROFILE_RIGIFY,
)
from .core.cache.stream import FrameSpool
from .core.errors import BackendError, InvalidRequest, MediaError
from .models import CanonicalFrame, Landmark
from .pose_translation_solver import PoseTranslationSolver
from .post_processing import (
    PostProcessingSession,
    SequenceProcessingContext,
    compile_postprocess_config,
    iter_pipeline,
    iter_sequence_pipeline,
    run_pipeline,
)
from .target_pose import (
    TargetPose,
)
from .target_pose import (
    signed_twist_angle as _signed_twist_angle,
)
from .target_pose import (
    stable_orientation_delta as _stable_orientation_delta,
)


@dataclass
class _BakedBoneSample:
    bone: object
    location: Vector
    rotation: Quaternion
    scale: Vector
    key_location: bool = False
    key_rotation: bool = False
    semantic_id: str = ""
    confidence: float = 1.0
    properties: dict[str, float] = field(default_factory=dict)


@dataclass
class _BakedShapeSample:
    block: object
    value: float


@dataclass
class _BakedFrameSample:
    destination: float
    warning_sink: list[str]
    timestamp_seconds: float = 0.0
    bones: dict[str, _BakedBoneSample] = field(default_factory=dict)
    shapes: dict[tuple[int, str], _BakedShapeSample] = field(default_factory=dict)
    root_object: tuple[Vector, Quaternion, Vector] | None = None
    grounding: grounded_cleanup.GroundFrame | None = None
    ik_legs: dict[str, dict] = field(default_factory=dict)
    body_height: float = 0.0
    camera_motion_input: camera_motion_correction.CameraMotionInput | None = None
    rohm_joint_offsets: dict[str, Vector] = field(default_factory=dict)
    rohm_rotations: dict[str, Quaternion] = field(default_factory=dict)
    rohm_confidence: dict[str, float] = field(default_factory=dict)
    rohm_root_confidence: float = 0.0
    source_body_3d: dict[str, Vector] = field(default_factory=dict)
    source_body_2d: dict[str, Vector] = field(default_factory=dict)


@dataclass
class _RohmFrameInput:
    """Pre-postprocess semantic motion in target-independent anatomy space."""

    joint_offsets: dict[str, Vector] = field(default_factory=dict)
    global_rotations: dict[str, Quaternion] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    body_height: float = 0.0


@dataclass(frozen=True)
class _RootMotionSample:
    timestamp_seconds: float
    observation: object
    source_basis: Matrix
    source_height: float


@dataclass
class _KeyframeWriteState:
    armature: object
    action: object = None
    previous_rotations: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    previous_object_rotation: list[float] | None = None
    total_bones: int = 0
    total_shapes: int = 0

    def __post_init__(self) -> None:
        if self.action is None:
            self.action = _ensure_action(self.armature)

    def attach(self) -> None:
        self.armature.animation_data_create().action = self.action

    def detach(self) -> None:
        animation = self.armature.animation_data
        if animation is not None and animation.action is self.action:
            animation.action = None


@dataclass(frozen=True)
class SequenceApplyProgress:
    phase: str
    current: int
    total: int


_PROFILE_OUTPUT_BONES = {
    PROFILE_ARP: {
        "SPINE_01": "spine_01.x",
        "SPINE_02": "spine_02.x",
        "CHEST": "spine_03.x",
        "NECK": "neck.x",
        "HEAD": "head.x",
        "UPPER_ARM_L": "arm.l",
        "LOWER_ARM_L": "forearm.l",
        "HAND_L": "hand.l",
        "UPPER_ARM_R": "arm.r",
        "LOWER_ARM_R": "forearm.r",
        "HAND_R": "hand.r",
        "UPPER_LEG_L": "thigh.l",
        "LOWER_LEG_L": "leg.l",
        "FOOT_L": "foot.l",
        "TOE_L": "toes_01.l",
        "UPPER_LEG_R": "thigh.r",
        "LOWER_LEG_R": "leg.r",
        "FOOT_R": "foot.r",
        "TOE_R": "toes_01.r",
        # ARP's first visible finger control is followed by a separate
        # weighted proximal deform bone. Later phalanges are themselves
        # deforming controls, so only segment 01 needs an output override.
        "THUMB_01_L": "thumb1.l",
        "INDEX_01_L": "index1.l",
        "MIDDLE_01_L": "middle1.l",
        "RING_01_L": "ring1.l",
        "PINKY_01_L": "pinky1.l",
        "THUMB_01_R": "thumb1.r",
        "INDEX_01_R": "index1.r",
        "MIDDLE_01_R": "middle1.r",
        "RING_01_R": "ring1.r",
        "PINKY_01_R": "pinky1.r",
    },
    PROFILE_RIGIFY: {
        "UPPER_ARM_L": "ORG-upper_arm.L",
        "LOWER_ARM_L": "ORG-forearm.L",
        "HAND_L": "ORG-hand.L",
        "UPPER_ARM_R": "ORG-upper_arm.R",
        "LOWER_ARM_R": "ORG-forearm.R",
        "HAND_R": "ORG-hand.R",
        "UPPER_LEG_L": "ORG-thigh.L",
        "LOWER_LEG_L": "ORG-shin.L",
        "FOOT_L": "ORG-foot.L",
        "TOE_L": "ORG-toe.L",
        "UPPER_LEG_R": "ORG-thigh.R",
        "LOWER_LEG_R": "ORG-shin.R",
        "FOOT_R": "ORG-foot.R",
        "TOE_R": "ORG-toe.R",
    },
}


def _average(
    *landmarks: Landmark | None,
    minimum_valid: int = 1,
) -> Landmark | None:
    valid = [item for item in landmarks if item and item.valid]
    if len(valid) < minimum_valid:
        return None
    size = len(valid)
    return Landmark(
        position=tuple(
            sum(item.position[i] for item in valid) / size for i in range(3)
        ),
        image=tuple(sum(item.image[i] for item in valid) / size for i in range(2)),
        confidence=min(item.confidence for item in valid),
        valid=True,
    )


def _parent_relative_direction_delta(
    rest_direction: Vector,
    desired_direction: Vector,
    parent_delta: Quaternion,
) -> Quaternion:
    """Transport parent roll, then add only the child's local swing."""
    local_direction = parent_delta.inverted() @ desired_direction
    articulation = rest_direction.rotation_difference(local_direction)
    return (parent_delta @ articulation).normalized()


def _bounded_parent_relative_direction_delta(
    rest_direction: Vector,
    desired_direction: Vector,
    parent_delta: Quaternion,
    maximum_articulation: float,
    *,
    bend_reference: Vector | None = None,
    maximum_off_plane: float = math.radians(25.0),
    flexion_bounds: tuple[float, float] | None = None,
) -> tuple[Quaternion, bool]:
    """Transport a parent, then add a bounded child-local swing.

    Landmark positions contain a reliable segment direction but no axial-roll
    observation. Solving each segment independently lets a single bad depth
    estimate turn a child almost 180 degrees even when its parent is sound.
    Expressing the direction after removing the effective parent transform
    makes the articulation local to the chain; clamping that shortest swing
    prevents an inside-out joint without restricting whole-arm motion.
    """
    if rest_direction.length <= 1.0e-8 or desired_direction.length <= 1.0e-8:
        return parent_delta.normalized(), True
    local_direction = (parent_delta.inverted() @ desired_direction).normalized()
    constrained = False
    if bend_reference is not None and bend_reference.length > 1.0e-8:
        rest_axis = rest_direction.normalized()
        flexion = bend_reference - rest_axis * bend_reference.dot(rest_axis)
        if flexion.length > 1.0e-8:
            flexion.normalize()
            lateral = rest_axis.cross(flexion).normalized()
            lateral_component = min(
                max(local_direction.dot(lateral), -1.0),
                1.0,
            )
            off_plane = math.asin(lateral_component)
            bounded_off_plane = min(
                max(off_plane, -maximum_off_plane),
                maximum_off_plane,
            )
            flexion_angle = math.atan2(
                local_direction.dot(flexion),
                local_direction.dot(rest_axis),
            )
            cosine_off_plane = max(math.cos(bounded_off_plane), 1.0e-8)
            maximum_flexion = math.acos(
                min(
                    max(
                        math.cos(maximum_articulation) / cosine_off_plane,
                        -1.0,
                    ),
                    1.0,
                )
            )
            minimum_flexion, maximum_flexion_bound = (
                flexion_bounds
                if flexion_bounds is not None
                else (-maximum_flexion, maximum_flexion)
            )
            minimum_flexion = max(float(minimum_flexion), -maximum_flexion)
            maximum_flexion_bound = min(float(maximum_flexion_bound), maximum_flexion)
            bounded_flexion = min(
                max(flexion_angle, minimum_flexion),
                maximum_flexion_bound,
            )
            constrained = bool(
                abs(bounded_off_plane - off_plane) > 1.0e-7
                or abs(bounded_flexion - flexion_angle) > 1.0e-7
            )
            local_direction = (
                rest_axis * math.cos(bounded_off_plane) * math.cos(bounded_flexion)
                + flexion * math.cos(bounded_off_plane) * math.sin(bounded_flexion)
                + lateral * math.sin(bounded_off_plane)
            ).normalized()
            articulation = rest_axis.rotation_difference(local_direction)
            return (parent_delta @ articulation).normalized(), constrained
    articulation = rest_direction.rotation_difference(local_direction)
    articulation, constrained = _clamp_rotation(
        articulation,
        maximum_articulation,
    )
    return (parent_delta @ articulation).normalized(), constrained


_FINGER_ARTICULATION_LIMITS = {
    "THUMB": tuple(math.radians(value) for value in (105.0, 115.0, 100.0)),
    "FINGER": tuple(math.radians(value) for value in (105.0, 130.0, 105.0)),
}


def body_points(frame: CanonicalFrame) -> dict[str, Landmark]:
    points = dict(frame.body)
    # Optional provider observations stay outside the canonical body mapping.
    # MHR70's NECK is a single surface/joint position, not a complete segment;
    # keep it available for diagnostics but never infer a neck-bone direction
    # from it and the shoulder midpoint.
    neck = frame.auxiliary.get("NECK")
    if neck is not None and neck.valid:
        points["NECK"] = neck
    points["HIP_MID"] = _average(
        points.get("LEFT_HIP"),
        points.get("RIGHT_HIP"),
        minimum_valid=2,
    )
    points["SHOULDER_MID"] = _average(
        points.get("LEFT_SHOULDER"),
        points.get("RIGHT_SHOULDER"),
        minimum_valid=2,
    )
    # The head bone points toward the cranium, not toward the front of the
    # face. Including NOSE here turns its anatomical forward offset into a
    # false neck-to-head lean for providers (notably MHR70) that expose an
    # actual neck joint.
    left_head = _average(points.get("LEFT_EAR"), points.get("LEFT_EYE"))
    right_head = _average(points.get("RIGHT_EAR"), points.get("RIGHT_EYE"))
    points["HEAD_MID"] = _average(left_head, right_head, minimum_valid=2)
    points["LEFT_HAND_MID"] = _average(
        points.get("LEFT_INDEX"),
        points.get("LEFT_PINKY"),
        points.get("LEFT_THUMB"),
        minimum_valid=2,
    )
    points["RIGHT_HAND_MID"] = _average(
        points.get("RIGHT_INDEX"),
        points.get("RIGHT_PINKY"),
        points.get("RIGHT_THUMB"),
        minimum_valid=2,
    )
    return {name: item for name, item in points.items() if item is not None}


def _canonical_vector(position: Sequence[float]) -> Vector:
    # Canonical providers emit image X, downward Y, and camera-relative depth,
    # with +Z pointing away from the camera. Convert that right-handed frame to
    # image-right/camera-away/up. The detected hip frame then removes source
    # viewpoint yaw and yields anatomical right/forward/up.
    return Vector((float(position[0]), float(position[2]), -float(position[1])))


def _mapped_bone(armature, entries, semantic_id: str):
    entry = entries.get(semantic_id)
    if not entry or not entry.enabled or not entry.target_name:
        return None
    return armature.pose.bones.get(entry.target_name)


def _profile_id(config) -> str:
    return mapping.profile_identifier(config)


def evaluated_semantic_bone(armature, entries, semantic_id: str):
    """Return the evaluated skeletal bone driven by a mapped control."""
    output_name = _PROFILE_OUTPUT_BONES.get(
        _profile_id(armature.mtp_rig_config), {}
    ).get(semantic_id)
    if output_name:
        output = armature.pose.bones.get(output_name)
        if output is not None:
            return output
    return _mapped_bone(armature, entries, semantic_id)


def normalized_arm_positions(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> tuple[dict[str, Vector], dict[str, Vector]]:
    """Build comparable detected and evaluated arm joint positions.

    The detected arm is normalized to the evaluated target shoulder position
    and to the target rig's upper-arm and forearm bone lengths. This
    removes provider scale, source placement, and armature object transforms,
    leaving a position-based check of the solved joint directions.
    """
    side = side.upper()
    if side not in {"L", "R"}:
        raise ValueError("Arm side must be 'L' or 'R'")
    label = "LEFT" if side == "L" else "RIGHT"
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    roles = (
        f"UPPER_ARM_{side}",
        f"LOWER_ARM_{side}",
    )
    points = body_points(frame)
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None or not start.valid or not end.valid:
            raise ValueError(
                f"Detected {label.lower()} arm is missing {start_name} or {end_name}"
            )
    bones = [evaluated_semantic_bone(armature, entries, role) for role in roles]
    if any(bone is None for bone in bones):
        raise ValueError(f"Evaluated {label.lower()} arm chain is incomplete")

    source_space = source_root_basis(frame)
    target_space = target_basis(armature, entries)
    directions = []
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        start = points.get(start_name)
        end = points.get(end_name)
        directions.append(
            _to_target_direction(
                Vector(end.position) - Vector(start.position),
                source_space,
                target_space,
            )
        )

    joint_names = (
        f"{label}_SHOULDER",
        f"{label}_ELBOW",
        f"{label}_WRIST",
    )
    detected = {joint_names[0]: bones[0].head.copy()}
    previous = detected[joint_names[0]]
    for joint_name, direction, bone in zip(
        joint_names[1:], directions, bones, strict=False
    ):
        previous = previous + direction.normalized() * float(bone.bone.length)
        detected[joint_name] = previous
    evaluated = {
        joint_names[0]: bones[0].head.copy(),
        joint_names[1]: bones[0].tail.copy(),
        joint_names[2]: bones[1].tail.copy(),
    }
    if world_space:
        matrix = armature.matrix_world
        detected = {name: matrix @ position for name, position in detected.items()}
        evaluated = {name: matrix @ position for name, position in evaluated.items()}
    return detected, evaluated


def normalized_arm_position_errors(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> dict[str, float]:
    """Return normalized joint-position errors for an evaluated arm pose."""
    detected, evaluated = normalized_arm_positions(
        armature,
        frame,
        side,
        world_space=world_space,
    )
    chain_length = 0.0
    for start_name, end_name in zip(tuple(detected), tuple(detected)[1:], strict=False):
        chain_length += (detected[end_name] - detected[start_name]).length
    if chain_length < 1.0e-8:
        raise ValueError("Evaluated arm chain has zero length")
    return {
        name: (evaluated[name] - detected[name]).length / chain_length
        for name in detected
    }


def normalized_leg_positions(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> tuple[dict[str, Vector], dict[str, Vector]]:
    """Build comparable normalized source and evaluated leg positions."""
    side = side.upper()
    if side not in {"L", "R"}:
        raise ValueError("Leg side must be 'L' or 'R'")
    label = "LEFT" if side == "L" else "RIGHT"
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    roles = (
        f"UPPER_LEG_{side}",
        f"LOWER_LEG_{side}",
    )
    points = body_points(frame)
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None or not start.valid or not end.valid:
            raise ValueError(
                f"Detected {label.lower()} leg is missing {start_name} or {end_name}"
            )
    bones = [evaluated_semantic_bone(armature, entries, role) for role in roles]
    if any(bone is None for bone in bones):
        raise ValueError(f"Evaluated {label.lower()} leg chain is incomplete")

    source_space = source_root_basis(frame)
    target_space = target_basis(armature, entries)
    directions = []
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        directions.append(
            _to_target_direction(
                Vector(points[end_name].position) - Vector(points[start_name].position),
                source_space,
                target_space,
            )
        )
    joint_names = (
        f"{label}_HIP",
        f"{label}_KNEE",
        f"{label}_ANKLE",
    )
    detected = {joint_names[0]: bones[0].head.copy()}
    previous = detected[joint_names[0]]
    for joint_name, direction, bone in zip(
        joint_names[1:], directions, bones, strict=False
    ):
        previous = previous + direction.normalized() * float(bone.bone.length)
        detected[joint_name] = previous
    evaluated = {
        joint_names[0]: bones[0].head.copy(),
        joint_names[1]: bones[0].tail.copy(),
        joint_names[2]: bones[1].tail.copy(),
    }
    if world_space:
        matrix = armature.matrix_world
        detected = {name: matrix @ position for name, position in detected.items()}
        evaluated = {name: matrix @ position for name, position in evaluated.items()}
    return detected, evaluated


def normalized_leg_position_errors(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> dict[str, float]:
    """Return normalized joint-position errors for an evaluated leg pose."""
    detected, evaluated = normalized_leg_positions(
        armature,
        frame,
        side,
        world_space=world_space,
    )
    names = tuple(detected)
    chain_length = sum(
        (detected[end] - detected[start]).length
        for start, end in zip(names, names[1:], strict=False)
    )
    if chain_length < 1.0e-8:
        raise ValueError("Evaluated leg chain has zero length")
    return {
        name: (evaluated[name] - detected[name]).length / chain_length
        for name in detected
    }


def normalized_head_position_errors(
    armature,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Compare evaluated cranial orientation with normalized detections.

    The diagnostic intentionally measures positions after dependency-graph
    evaluation.  Source placement and scale are removed by anchoring each
    expected endpoint at the target bone head and using that target segment's
    length.
    """
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    points = body_points(frame)
    head_mid = points.get("HEAD_MID")
    if head_mid is None or not head_mid.valid:
        raise ValueError("Detected head center is missing")
    source_space = source_root_basis(frame)
    target_space = target_basis(armature, entries)
    errors: dict[str, float] = {}

    head = evaluated_semantic_bone(armature, entries, "HEAD")
    # The evaluated pose uses the compact MHR127 skull frame whenever SAM
    # supplies it.  Compare against that same authoritative observation;
    # falling back to the sparse eye/ear frame here reported a large false
    # error even when retargeting reproduced the MHR head exactly.
    orientation = _source_mhr_head_orientation(frame) or _source_head_orientation(
        points
    )
    if head is None or orientation is None:
        raise ValueError("Evaluated head or detected head frame is missing")
    head_length = max(float(head.length), 1.0e-8)
    expected_up = _to_target_direction(orientation[0], source_space, target_space)
    expected_tip = head.head + expected_up * head_length
    errors["HEAD_TIP"] = (head.tail - expected_tip).length / head_length

    expected_lateral = _to_target_direction(orientation[1], source_space, target_space)
    rest_lateral = _target_body_reference(armature, entries, "HEAD")
    if (
        rest_lateral is not None
        and rest_lateral.length > 1.0e-8
        and expected_lateral.length > 1.0e-8
    ):
        head_delta = (
            head.matrix.to_quaternion()
            @ head.bone.matrix_local.to_quaternion().inverted()
        )
        evaluated_lateral = head_delta @ rest_lateral.normalized()
        expected_side = head.head + expected_lateral.normalized() * head_length
        evaluated_side = head.head + evaluated_lateral.normalized() * head_length
        errors["HEAD_LATERAL"] = (evaluated_side - expected_side).length / head_length

    return errors


def _root_frame(right: Vector, up: Vector) -> Matrix:
    """Build a proper right/forward/up rotation frame."""
    up = up.normalized() if up.length > 1.0e-8 else Vector((0.0, 0.0, 1.0))
    right = right - up * right.dot(up)
    if right.length < 1.0e-8:
        right = Vector((-1.0, 0.0, 0.0))
        right = right - up * right.dot(up)
    right.normalize()
    forward = up.cross(right)
    if forward.length < 1.0e-8:
        forward = Vector((0.0, -1.0, 0.0))
    else:
        forward.normalize()
    up = right.cross(forward).normalized()
    return Matrix((right, forward, up)).transposed()


def source_root_basis(frame: CanonicalFrame) -> Matrix:
    """Return the detected person's yaw frame in canonical source space."""
    points = body_points(frame)
    lateral_vectors = []
    for left_name, right_name in (
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ):
        left = points.get(left_name)
        right = points.get(right_name)
        if not left or not right or not left.valid or not right.valid:
            continue
        lateral = _canonical_vector(Vector(right.position) - Vector(left.position))
        # Use detected depth for subject yaw, while image vertical remains the
        # stable up axis. Body lean stays in the individual pose vectors.
        lateral.z = 0.0
        if lateral.length > 1.0e-8:
            lateral_vectors.append(lateral.normalized())

    if lateral_vectors:
        # Hips and shoulders estimate the same anatomical axis. Averaging them
        # damps monocular depth jitter without involving any Blender camera.
        right = sum(lateral_vectors, Vector())
        if right.length < 1.0e-8:
            right = lateral_vectors[0]
    else:
        right = Vector((-1.0, 0.0, 0.0))
    return _root_frame(right, Vector((0.0, 0.0, 1.0)))


def target_rest_basis(armature, entries) -> Matrix:
    """Return the immutable character rest frame in armature-local space."""
    pelvis = _mapped_bone(armature, entries, "PELVIS")
    chest = _mapped_bone(armature, entries, "CHEST")
    left_leg = _mapped_bone(armature, entries, "UPPER_LEG_L")
    right_leg = _mapped_bone(armature, entries, "UPPER_LEG_R")
    left_arm = _mapped_bone(armature, entries, "UPPER_ARM_L")
    right_arm = _mapped_bone(armature, entries, "UPPER_ARM_R")

    def head(bone):
        return bone.bone.head_local.copy() if bone else None

    left_hip = head(left_leg)
    right_hip = head(right_leg)
    left_shoulder = head(left_arm)
    right_shoulder = head(right_arm)
    right = (
        right_hip - left_hip
        if left_hip is not None and right_hip is not None
        else Vector((-1.0, 0.0, 0.0))
    )
    if (
        left_hip is not None
        and right_hip is not None
        and left_shoulder is not None
        and right_shoulder is not None
    ):
        hip_center = (left_hip + right_hip) * 0.5
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        up = shoulder_center - hip_center
    else:
        center = head(pelvis)
        upper = head(chest)
        up = (
            upper - center
            if center is not None and upper is not None
            else Vector((0.0, 0.0, 1.0))
        )
    return _root_frame(right, up)


def target_basis(
    armature,
    entries,
    *,
    rest_basis: Matrix | None = None,
) -> Matrix:
    """Return the character root frame including its current root control."""
    rest_frame = (
        rest_basis.copy()
        if rest_basis is not None
        else target_rest_basis(armature, entries)
    )

    # The armature object transform already stays outside pose space. Include
    # the mapped root control's current rotation so the capture is relative to
    # the character placement, without ever consulting a Blender camera.
    root = _mapped_bone(armature, entries, "ROOT")
    if root:
        # Extract rotation only. Pose/control scale must not shear the
        # camera-independent retarget frame.
        root_delta = (
            root.matrix.to_quaternion().to_matrix()
            @ root.bone.matrix_local.to_quaternion().inverted().to_matrix()
        )
        return root_delta @ rest_frame
    return rest_frame


def _to_target_direction(
    raw_direction: Vector,
    source_basis: Matrix,
    target_basis: Matrix,
) -> Vector:
    canonical = _canonical_vector(raw_direction)
    root_relative = source_basis.inverted_safe() @ canonical
    result = target_basis @ root_relative
    return result.normalized() if result.length > 1.0e-8 else result


def _joint_frame_target_delta(
    frame: CanonicalFrame,
    name: str,
    source_basis: Matrix,
    target_basis: Matrix,
) -> Quaternion | None:
    """Express a neutral-relative provider joint rotation in target space."""
    observation = frame.joint_frames.get(name)
    if observation is None or not observation.valid:
        return None
    values = observation.rotation_matrix
    if len(values) != 9 or not all(math.isfinite(float(value)) for value in values):
        return None
    raw_delta = Matrix(
        tuple(
            tuple(float(values[row * 3 + column]) for column in range(3))
            for row in range(3)
        )
    )
    # Provider positions use image-right, image-down, camera-depth. Convert
    # the rotation itself through the same basis change as point directions,
    # then express its axis relative to the detected and target root frames.
    canonical = Matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
    )
    canonical_delta = canonical @ raw_delta @ canonical.inverted_safe()
    root_relative = source_basis.inverted_safe() @ canonical_delta @ source_basis
    target_delta = target_basis @ root_relative @ target_basis.inverted_safe()
    result = target_delta.to_quaternion().normalized()
    if result.w < 0.0:
        result.negate()
    return result


def _clamp_rotation(
    rotation: Quaternion, maximum_angle: float
) -> tuple[Quaternion, bool]:
    """Limit a shortest-arc rotation without changing its axis."""
    rotation = rotation.normalized()
    if rotation.w < 0.0:
        rotation.negate()
    angle = rotation.angle
    if angle <= maximum_angle:
        return rotation, False
    return Quaternion().slerp(rotation, maximum_angle / angle).normalized(), True


_BODY_CHILD_ROLE = {
    "SPINE_01": "SPINE_02",
    "SPINE_02": "CHEST",
    "CHEST": "NECK",
    "NECK": "HEAD",
    "SHOULDER_L": "UPPER_ARM_L",
    "UPPER_ARM_L": "LOWER_ARM_L",
    "LOWER_ARM_L": "HAND_L",
    "SHOULDER_R": "UPPER_ARM_R",
    "UPPER_ARM_R": "LOWER_ARM_R",
    "LOWER_ARM_R": "HAND_R",
    "UPPER_LEG_L": "LOWER_LEG_L",
    "LOWER_LEG_L": "FOOT_L",
    "FOOT_L": "TOE_L",
    "UPPER_LEG_R": "LOWER_LEG_R",
    "LOWER_LEG_R": "FOOT_R",
    "FOOT_R": "TOE_R",
}


def _rest_head(armature, entries, semantic_id: str) -> Vector | None:
    bone = _mapped_bone(armature, entries, semantic_id)
    return bone.bone.head_local.copy() if bone else None


def _finger_role(finger: str, segment: int, side: str) -> str:
    return f"{finger}_{segment:02d}_{side}"


def _target_palm_vectors(
    armature, entries, side: str
) -> tuple[Vector | None, Vector | None]:
    hand = _rest_head(armature, entries, f"HAND_{side}")
    finger_heads = [
        _rest_head(armature, entries, _finger_role(finger, 1, side))
        for finger in ("INDEX", "MIDDLE", "RING", "PINKY")
    ]
    valid_heads = [point for point in finger_heads if point is not None]
    primary = (
        sum(valid_heads, Vector()) / len(valid_heads) - hand
        if hand is not None and valid_heads
        else None
    )
    index = _rest_head(armature, entries, _finger_role("INDEX", 1, side))
    pinky = _rest_head(armature, entries, _finger_role("PINKY", 1, side))
    lateral = index - pinky if index is not None and pinky is not None else None
    return primary, lateral


def _orthonormal_palm_basis(
    primary: Vector | None,
    lateral: Vector | None,
) -> Matrix | None:
    """Return a right-handed palm-local frame with axes stored as columns."""
    if (
        primary is None
        or lateral is None
        or primary.length <= 1.0e-8
        or lateral.length <= 1.0e-8
    ):
        return None
    forward = primary.normalized()
    across = lateral - forward * lateral.dot(forward)
    if across.length <= 1.0e-8:
        return None
    across.normalize()
    normal = forward.cross(across)
    if normal.length <= 1.0e-8:
        return None
    normal.normalize()
    return Matrix((forward, across, normal)).transposed()


def _detected_palm_basis(hand) -> Matrix | None:
    wrist = hand.get("WRIST")
    palm_points = [
        hand.get("INDEX_FINGER_MCP"),
        hand.get("MIDDLE_FINGER_MCP"),
        hand.get("RING_FINGER_MCP"),
        hand.get("PINKY_MCP"),
    ]
    valid_palm = [point for point in palm_points if point and point.valid]
    index = hand.get("INDEX_FINGER_MCP")
    pinky = hand.get("PINKY_MCP")
    if (
        wrist is None
        or not wrist.valid
        or not valid_palm
        or index is None
        or not index.valid
        or pinky is None
        or not pinky.valid
    ):
        return None
    palm_center = sum(
        (Vector(point.position) for point in valid_palm),
        Vector(),
    ) / len(valid_palm)
    return _orthonormal_palm_basis(
        palm_center - Vector(wrist.position),
        Vector(index.position) - Vector(pinky.position),
    )


def _target_body_direction(armature, entries, semantic_id: str) -> Vector | None:
    start = _rest_head(armature, entries, semantic_id)
    child_id = _BODY_CHILD_ROLE.get(semantic_id)
    end = _rest_head(armature, entries, child_id) if child_id else None
    if start is not None and end is not None and (end - start).length > 1.0e-8:
        return end - start
    if semantic_id in {"HAND_L", "HAND_R"}:
        primary, _lateral = _target_palm_vectors(armature, entries, semantic_id[-1])
        if primary is not None and primary.length > 1.0e-8:
            return primary
    bone = _mapped_bone(armature, entries, semantic_id)
    if bone:
        return bone.bone.tail_local - bone.bone.head_local
    return None


def _target_torso_vectors(armature, entries) -> tuple[Vector | None, Vector | None]:
    left = _rest_head(armature, entries, "UPPER_ARM_L")
    right = _rest_head(armature, entries, "UPPER_ARM_R")
    pelvis = _rest_head(armature, entries, "PELVIS")
    chest = _rest_head(armature, entries, "CHEST")
    lateral = right - left if left is not None and right is not None else None
    up = chest - pelvis if pelvis is not None and chest is not None else None
    return lateral, up


def _target_body_reference(armature, entries, semantic_id: str) -> Vector | None:
    lateral, up = _target_torso_vectors(armature, entries)
    if semantic_id in {
        "SPINE_01",
        "SPINE_02",
        "CHEST",
        "NECK",
        "HEAD",
    }:
        return lateral
    if semantic_id.startswith("SHOULDER_"):
        return up
    adjacent = {
        "UPPER_ARM_L": "LOWER_ARM_L",
        "LOWER_ARM_L": "UPPER_ARM_L",
        "UPPER_ARM_R": "LOWER_ARM_R",
        "LOWER_ARM_R": "UPPER_ARM_R",
        "UPPER_LEG_L": "LOWER_LEG_L",
        "LOWER_LEG_L": "UPPER_LEG_L",
        "FOOT_L": "LOWER_LEG_L",
        "TOE_L": "LOWER_LEG_L",
        "UPPER_LEG_R": "LOWER_LEG_R",
        "LOWER_LEG_R": "UPPER_LEG_R",
        "FOOT_R": "LOWER_LEG_R",
        "TOE_R": "LOWER_LEG_R",
    }.get(semantic_id)
    if adjacent:
        return _target_body_direction(armature, entries, adjacent)
    if semantic_id in {"HAND_L", "HAND_R"}:
        _primary, palm_lateral = _target_palm_vectors(
            armature, entries, semantic_id[-1]
        )
        return palm_lateral
    return None


def _source_direction(points, start: str, end: str) -> Vector | None:
    first, second = points.get(start), points.get(end)
    if not first or not second or not first.valid or not second.valid:
        return None
    direction = Vector(second.position) - Vector(first.position)
    return direction if direction.length > 1.0e-8 else None


def _source_body_reference(points, semantic_id: str) -> Vector | None:
    if semantic_id in {
        "SPINE_01",
        "SPINE_02",
        "CHEST",
        "NECK",
        "HEAD",
    }:
        return _source_direction(points, "LEFT_SHOULDER", "RIGHT_SHOULDER")
    if semantic_id.startswith("SHOULDER_"):
        return _source_direction(points, "HIP_MID", "SHOULDER_MID")
    reference_pairs = {
        "UPPER_ARM_L": ("LEFT_ELBOW", "LEFT_WRIST"),
        "LOWER_ARM_L": ("LEFT_SHOULDER", "LEFT_ELBOW"),
        "HAND_L": ("LEFT_PINKY", "LEFT_INDEX"),
        "UPPER_ARM_R": ("RIGHT_ELBOW", "RIGHT_WRIST"),
        "LOWER_ARM_R": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        "HAND_R": ("RIGHT_PINKY", "RIGHT_INDEX"),
        "UPPER_LEG_L": ("LEFT_KNEE", "LEFT_ANKLE"),
        "LOWER_LEG_L": ("LEFT_HIP", "LEFT_KNEE"),
        "FOOT_L": ("LEFT_KNEE", "LEFT_ANKLE"),
        "TOE_L": ("LEFT_KNEE", "LEFT_ANKLE"),
        "UPPER_LEG_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
        "LOWER_LEG_R": ("RIGHT_HIP", "RIGHT_KNEE"),
        "FOOT_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
        "TOE_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
    }
    pair = reference_pairs.get(semantic_id)
    return _source_direction(points, *pair) if pair else None


def _source_head_orientation(
    points,
) -> tuple[Vector, Vector] | None:
    """Return provider-space head-up and lateral vectors from positions.

    Head roll and yaw are observable from the relative eye/ear positions.
    A provider neck joint is useful for the neck segment, but it is not a
    valid start point for a Blender head bone: eyes and ears sit physically
    forward of the cervical axis, and MHR70 has no skull-tip joint to form a
    true head segment. Use the image-plane face axis for head up instead,
    avoiding invented nod from this anatomical offset or shallow face depth.
    Explicit provider joint rotations are never copied.
    """
    left_ear = points.get("LEFT_EAR")
    right_ear = points.get("RIGHT_EAR")
    left_eye = points.get("LEFT_EYE")
    right_eye = points.get("RIGHT_EYE")
    shoulders = points.get("SHOULDER_MID")
    head_mid = points.get("HEAD_MID")
    lateral_samples = []
    for left, right in (
        (left_ear, right_ear),
        (left_eye, right_eye),
    ):
        if left is None or right is None or not left.valid or not right.valid:
            continue
        sample = Vector(right.position) - Vector(left.position)
        if sample.length > 1.0e-8:
            sample.normalize()
            if lateral_samples and sample.dot(lateral_samples[0]) < 0.0:
                sample.negate()
            lateral_samples.append(sample)
    if not lateral_samples:
        return None
    lateral = sum(lateral_samples, Vector())
    if lateral.length < 1.0e-8:
        return None
    lateral.normalize()

    # Provider coordinates are image-right X, image-down Y, and depth Z.
    # Rotate the observed face lateral 90 degrees in the image plane. The full
    # lateral (including depth) still carries face yaw into the two-vector
    # orientation solve, but no invalid neck-to-face segment can invent nod.
    up = Vector((lateral.y, -lateral.x, 0.0))
    if lateral.length < 1.0e-8 or up.length < 1.0e-8:
        return None
    # Face landmark ordering can flip the cross-product sign across provider
    # conventions. Align it to the observed torso-to-head direction.
    if shoulders and head_mid and shoulders.valid and head_mid.valid:
        torso_up = Vector(head_mid.position) - Vector(shoulders.position)
        if (
            torso_up.length > 1.0e-8
            and _canonical_vector(up).dot(_canonical_vector(torso_up)) < 0.0
        ):
            up.negate()
    return up, lateral


def _source_mhr_head_orientation(
    frame: CanonicalFrame,
) -> tuple[Vector, Vector] | None:
    """Build the SAM skull-up frame from articulated MHR127 positions.

    ``c_head_null`` is the end-site extension parented to ``c_head``.  Its
    position therefore supplies the head bone's primary (anatomical-up) axis;
    the eye centres supply the lateral axis.  Do not derive up by crossing the
    eye line with this vector: doing so treats the end site as a face-forward
    marker and folds an otherwise upright skull by roughly 90 degrees.
    """
    head = frame.auxiliary.get("MHR_HEAD")
    tip = frame.auxiliary.get("MHR_HEAD_TIP")
    left_eye = frame.auxiliary.get("MHR_EYE_L")
    right_eye = frame.auxiliary.get("MHR_EYE_R")
    if not all(
        item is not None and item.valid for item in (head, tip, left_eye, right_eye)
    ):
        return None
    primary = Vector(tip.position) - Vector(head.position)
    lateral = Vector(right_eye.position) - Vector(left_eye.position)
    if primary.length < 1.0e-6 or lateral.length < 1.0e-6:
        return None
    # On a near-frontal face, monocular crown depth is weakly constrained and
    # stylized SAM fits can turn a modest head roll into a pronounced nod.
    # Keep the observed image-plane crown direction authoritative and cap only
    # its camera-depth component. Profile views retain the full MHR depth.
    body_left_eye = frame.body.get("LEFT_EYE")
    body_right_eye = frame.body.get("RIGHT_EYE")
    body_left_shoulder = frame.body.get("LEFT_SHOULDER")
    body_right_shoulder = frame.body.get("RIGHT_SHOULDER")
    image_frontal = False
    if all(
        item is not None and item.valid
        for item in (
            body_left_eye,
            body_right_eye,
            body_left_shoulder,
            body_right_shoulder,
        )
    ):
        eye_width = math.dist(body_left_eye.image, body_right_eye.image)
        shoulder_width = math.dist(body_left_shoulder.image, body_right_shoulder.image)
        image_frontal = shoulder_width > 1.0e-6 and eye_width / shoulder_width > 0.25
    if image_frontal or abs(lateral.z) / lateral.length < 0.35:
        planar_length = math.hypot(primary.x, primary.y)
        if planar_length > 1.0e-6:
            depth_limit = planar_length * math.tan(math.radians(15.0))
            primary.z = min(max(primary.z, -depth_limit), depth_limit)
    axis = primary.normalized()
    projected = lateral - axis * lateral.dot(axis)
    if projected.length < lateral.length * 0.12:
        return None
    neck = frame.auxiliary.get("MHR_NECK")
    if neck and neck.valid:
        cervical = Vector(head.position) - Vector(neck.position)
        if cervical.length > 1.0e-6 and primary.dot(cervical) < 0.0:
            primary.negate()
    return primary, projected


def _apply_body(
    armature,
    entries,
    frame,
    source_basis,
    target_basis,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    changed = set()
    points = body_points(frame)
    # Root and pelvis are placement controls and remain unchanged. Segment
    # directions are solved between source and target root frames.
    for semantic_id, pair in BODY_JOINT_PAIRS.items():
        entry = entries.get(semantic_id)
        bone = _mapped_bone(armature, entries, semantic_id)
        start, end = points.get(pair[0]), points.get(pair[1])
        if semantic_id == "NECK":
            if not entry or not bone:
                continue
            # MHR_HEAD sits forward of the cervical axis even in the neutral
            # body. Aiming the neck at that absolute joint chord turns a
            # permanent anatomical offset into severe flexion. SAM already
            # supplies a neutral-relative neck frame, so use that articulated
            # observation and leave the head solve to eyes, ears, and crown.
            neck_delta = _joint_frame_target_delta(
                frame,
                "MHR_NECK",
                source_basis,
                target_basis,
            )
            if neck_delta is None:
                target_pose.set_basis_rotation(
                    semantic_id,
                    bone,
                    Quaternion(),
                    correction=entry.correction_euler,
                    owner="RETARGET",
                )
            else:
                neck_delta, clamped = _clamp_rotation(
                    neck_delta,
                    math.radians(45.0),
                )
                rest_direction = _target_body_direction(
                    armature,
                    entries,
                    semantic_id,
                )
                rest_reference = _target_body_reference(
                    armature,
                    entries,
                    semantic_id,
                )
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    neck_delta.to_matrix() @ bone.bone.matrix_local.to_3x3(),
                    correction=entry.correction_euler,
                    confidence=frame.joint_frames["MHR_NECK"].confidence,
                    influence=entry.influence,
                )
                channel = target_pose.channel_for(semantic_id)
                channel.rest_direction = rest_direction.copy()
                channel.desired_direction = neck_delta @ rest_direction
                channel.rest_reference = (
                    rest_reference.copy() if rest_reference is not None else None
                )
                channel.desired_reference = (
                    neck_delta @ rest_reference if rest_reference is not None else None
                )
                if clamped:
                    target_pose.warnings.append(
                        "SAM neck rotation was limited to 45 degrees"
                    )
            changed.add(bone)
            continue
        if (
            not entry
            or not bone
            or not start
            or not end
            or not start.valid
            or not end.valid
        ):
            continue
        raw_direction = Vector(end.position) - Vector(start.position)
        # A recovered MediaPipe hinge can have a usable image location while
        # its regressed world-space adjacent segment points in the opposite
        # direction (a common foreshortening failure on folded stylized arms).
        # When a recovered hinge starts a child segment, reconstruct that
        # segment from its image displacement. A very short projected forearm
        # relative to the projected upper arm is foreshortening, not a short
        # target bone: use the mapped chain's rest-length ratio to put the
        # missing component toward the camera. This keeps the wrist near the
        # observed elbow in projection without stretching the target limb.
        recovered_hinge = frame.auxiliary.get(f"RECOVERED_{pair[0]}") is not None
        if recovered_hinge:
            image_delta = Vector(
                (
                    float(end.image[0]) - float(start.image[0]),
                    float(end.image[1]) - float(start.image[1]),
                )
            )
            parent_role = {
                "LOWER_ARM_L": "UPPER_ARM_L",
                "LOWER_ARM_R": "UPPER_ARM_R",
                "LOWER_LEG_L": "UPPER_LEG_L",
                "LOWER_LEG_R": "UPPER_LEG_R",
            }.get(semantic_id)
            parent_pair = BODY_JOINT_PAIRS.get(parent_role, ())
            parent_start = points.get(parent_pair[0]) if parent_pair else None
            parent_bone = _mapped_bone(armature, entries, parent_role)
            parent_image_delta = (
                Vector(
                    (
                        float(start.image[0]) - float(parent_start.image[0]),
                        float(start.image[1]) - float(parent_start.image[1]),
                    )
                )
                if parent_start is not None
                else Vector()
            )
            if (
                image_delta.length > 1.0e-6
                and parent_image_delta.length > 1.0e-6
                and parent_bone is not None
                and bone.bone.length > 1.0e-8
            ):
                projected_fraction = min(
                    max(
                        image_delta.length
                        / parent_image_delta.length
                        * parent_bone.bone.length
                        / bone.bone.length,
                        0.0,
                    ),
                    1.0,
                )
                planar = image_delta.normalized() * projected_fraction
                image_direction = Vector(
                    (
                        planar.x,
                        planar.y,
                        -math.sqrt(max(1.0 - projected_fraction**2, 0.0)),
                    )
                )
                if image_direction.length > 1.0e-6:
                    raw_direction = image_direction
                    warning = (
                        f"{semantic_id.replace('_', ' ').title()} used "
                        "image-plane swing for a recovered hinge"
                    )
                    if warning not in target_pose.warnings:
                        target_pose.warnings.append(warning)
        source_reference = _source_body_reference(points, semantic_id)
        if semantic_id == "HEAD":
            head_orientation = _source_mhr_head_orientation(frame)
            if head_orientation is None:
                head_orientation = _source_head_orientation(points)
            if head_orientation is not None:
                raw_direction, source_reference = head_orientation
            else:
                target_pose.warnings.append(
                    "Head orientation was unavailable; existing pose was preserved"
                )
                continue
        direction = _to_target_direction(
            raw_direction,
            source_basis,
            target_basis,
        )
        desired_reference = (
            _to_target_direction(source_reference, source_basis, target_basis)
            if source_reference is not None
            else None
        )
        confidence = min(start.confidence, end.confidence)
        if _source_body_reference(points, semantic_id) is not None:
            reference_names = {
                "UPPER_ARM_L": ("LEFT_ELBOW", "LEFT_WRIST"),
                "LOWER_ARM_L": ("LEFT_SHOULDER", "LEFT_ELBOW"),
                "UPPER_ARM_R": ("RIGHT_ELBOW", "RIGHT_WRIST"),
                "LOWER_ARM_R": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
                "UPPER_LEG_L": ("LEFT_KNEE", "LEFT_ANKLE"),
                "LOWER_LEG_L": ("LEFT_HIP", "LEFT_KNEE"),
                "UPPER_LEG_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
                "LOWER_LEG_R": ("RIGHT_HIP", "RIGHT_KNEE"),
            }.get(semantic_id)
            if reference_names:
                reference_points = [points.get(name) for name in reference_names]
                confidence = min(
                    confidence,
                    *(
                        point.confidence
                        for point in reference_points
                        if point is not None
                    ),
                )
        # A toe has no independent lateral landmark. Its knee-to-ankle
        # reference is an unoriented bend-plane axis, and solving that axis
        # independently for both foot and toe can put the two adjacent
        # segments on opposite roll hemispheres near 90 degrees. Preserve the
        # already-solved foot roll and add only the toe's observed swing.
        # This keeps a small anatomical toe bend small instead of occasionally
        # turning it into a 180-degree axial twist.
        parent_channel = (
            target_pose.channel_for(f"FOOT_{semantic_id[-1]}")
            if semantic_id.startswith("TOE_")
            else None
        )
        if parent_channel is not None and parent_channel.desired_rotation is not None:
            parent_rest = parent_channel.bone.bone.matrix_local.to_quaternion()
            parent_delta = (
                parent_channel.desired_rotation.to_quaternion() @ parent_rest.inverted()
            ).normalized()
            rest_direction = _target_body_direction(armature, entries, semantic_id)
            toe_delta = _parent_relative_direction_delta(
                rest_direction,
                direction,
                parent_delta,
            )
            target_pose.set_desired_rotation(
                semantic_id,
                bone,
                toe_delta.to_matrix() @ bone.bone.matrix_local.to_3x3(),
                correction=entry.correction_euler,
                confidence=confidence,
                influence=entry.influence,
            )
            channel = target_pose.channel_for(semantic_id)
            channel.rest_direction = rest_direction.copy()
            channel.desired_direction = direction.copy()
            rest_reference = _target_body_reference(armature, entries, semantic_id)
            channel.rest_reference = (
                rest_reference.copy() if rest_reference is not None else None
            )
            channel.desired_reference = (
                desired_reference.copy() if desired_reference is not None else None
            )
            changed.add(bone)
            continue
        rest_direction = _target_body_direction(armature, entries, semantic_id)
        rest_reference = _target_body_reference(armature, entries, semantic_id)
        if semantic_id in {"HAND_L", "HAND_R"}:
            # Preserve the observed global palm frame. Forearm roll is assigned
            # later by Arm Axis Correction; clamping the wrist against the
            # still-untwisted forearm here made a smooth palm cross between
            # opposite 110-degree clamp branches.
            full_delta = _stable_orientation_delta(
                rest_direction,
                direction,
                rest_reference,
                desired_reference,
                reference_is_oriented=True,
            )
            if full_delta is not None:
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    full_delta.to_matrix() @ bone.bone.matrix_local.to_3x3(),
                    correction=entry.correction_euler,
                    confidence=confidence,
                    influence=entry.influence,
                )
                channel = target_pose.channel_for(semantic_id)
                channel.rest_direction = rest_direction.copy()
                channel.desired_direction = direction.copy()
                channel.rest_reference = (
                    rest_reference.copy() if rest_reference is not None else None
                )
                channel.desired_reference = (
                    desired_reference.copy() if desired_reference is not None else None
                )
                changed.add(bone)
                continue
        if target_pose.aim(
            semantic_id,
            bone,
            direction,
            rest_direction=rest_direction,
            desired_reference=desired_reference,
            rest_reference=rest_reference,
            correction=entry.correction_euler,
            confidence=confidence,
            influence=entry.influence,
            reference_is_oriented=semantic_id in {"HAND_L", "HAND_R"},
        ):
            changed.add(bone)
    return changed


def _apply_hands(
    armature,
    entries,
    frame,
    source_basis,
    target_basis,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    changed = set()
    profile = mapping.profile_identifier(armature.mtp_rig_config)
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name)
        if not hand:
            continue
        if profile == PROFILE_ARP:
            # ARP inserts a freely rotatable `c_*1_base` control above every
            # first phalanx. Reset it only when at least one valid, mapped
            # segment of that same finger will own the captured motion. A
            # visible wrist or unrelated finger must not modify the control.
            owned_fingers = set()
            for role_base, pair in HAND_JOINT_PAIRS.items():
                semantic_id = f"{role_base}_{side_suffix}"
                entry = entries.get(semantic_id)
                start, end = hand.get(pair[0]), hand.get(pair[1])
                if (
                    entry
                    and _mapped_bone(armature, entries, semantic_id)
                    and start
                    and end
                    and start.valid
                    and end.valid
                ):
                    owned_fingers.add(role_base.split("_", 1)[0].lower())
            for finger in owned_fingers:
                base = armature.pose.bones.get(
                    f"c_{finger}1_base.{side_suffix.lower()}"
                )
                if base is not None:
                    target_pose.set_basis_rotation(
                        "",
                        base,
                        Quaternion(),
                        owner="RETARGET",
                        priority=55,
                    )
                    changed.add(base)
        hand_id = f"HAND_{side_suffix}"
        hand_channel = target_pose.channel_for(hand_id)
        hand_delta = Quaternion()
        if hand_channel is not None and hand_channel.desired_rotation is not None:
            rest_hand_rotation = hand_channel.bone.bone.matrix_local.to_3x3()
            hand_delta = (
                (hand_channel.desired_rotation @ rest_hand_rotation.inverted_safe())
                .to_quaternion()
                .normalized()
            )
        palm_primary, palm_lateral = _target_palm_vectors(
            armature, entries, side_suffix
        )
        target_palm_basis = _orthonormal_palm_basis(
            palm_primary,
            palm_lateral,
        )
        detected_palm_basis = _detected_palm_basis(hand)
        palm_normal = (
            palm_primary.cross(palm_lateral).normalized()
            if palm_primary is not None
            and palm_lateral is not None
            and palm_primary.cross(palm_lateral).length > 1.0e-8
            else None
        )

        # Detailed landmarks own only articulated fingers. Every provider's
        # main hand bone remains owned by the body solver, so two independent
        # palm coordinate systems can no longer overwrite each other.
        for finger in ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"):
            parent_delta = hand_delta
            limits = _FINGER_ARTICULATION_LIMITS[
                "THUMB" if finger == "THUMB" else "FINGER"
            ]
            for segment in range(1, 4):
                role_base = f"{finger}_{segment:02d}"
                pair = HAND_JOINT_PAIRS[role_base]
                semantic_id = f"{role_base}_{side_suffix}"
                entry = entries.get(semantic_id)
                bone = _mapped_bone(armature, entries, semantic_id)
                start, end = hand.get(pair[0]), hand.get(pair[1])
                if (
                    not entry
                    or not bone
                    or not start
                    or not end
                    or not start.valid
                    or not end.valid
                ):
                    break
                raw_direction = Vector(end.position) - Vector(start.position)
                if detected_palm_basis is not None and target_palm_basis is not None:
                    # Finger observations are meaningful relative to their
                    # own palm, irrespective of which provider produced the
                    # body.  Mapping SAM/MHR hand points through the body's
                    # root frame can reflect their palm-normal component when
                    # the body wrist and detailed palm frames differ.  Keep
                    # the articulated chain provider-neutral and let the body
                    # solver's hand delta place that palm on the character.
                    palm_local = detected_palm_basis.inverted_safe() @ raw_direction
                    direction = hand_delta @ (target_palm_basis @ palm_local)
                    if direction.length > 1.0e-8:
                        direction.normalize()
                else:
                    direction = _to_target_direction(
                        raw_direction,
                        source_basis,
                        target_basis,
                    )
                if direction.length <= 1.0e-8:
                    break
                target_start = _rest_head(
                    armature, entries, _finger_role(finger, segment, side_suffix)
                )
                target_end = (
                    _rest_head(
                        armature,
                        entries,
                        _finger_role(finger, segment + 1, side_suffix),
                    )
                    if segment < 3
                    else None
                )
                target_direction = (
                    target_end - target_start
                    if target_start is not None and target_end is not None
                    else bone.bone.tail_local - bone.bone.head_local
                )
                finger_delta, clamped = _bounded_parent_relative_direction_delta(
                    target_direction,
                    direction,
                    parent_delta,
                    limits[segment - 1],
                    bend_reference=(palm_normal if finger != "THUMB" else None),
                    maximum_off_plane=math.radians(35.0 if segment == 1 else 25.0),
                    flexion_bounds=(
                        (
                            math.radians(-25.0 if segment == 1 else -15.0),
                            limits[segment - 1],
                        )
                        if side_suffix == "L"
                        else (
                            -limits[segment - 1],
                            math.radians(25.0 if segment == 1 else 15.0),
                        )
                    )
                    if finger != "THUMB"
                    else None,
                )
                rest_rotation = bone.bone.matrix_local.to_3x3()
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    finger_delta.to_matrix() @ rest_rotation,
                    correction=entry.correction_euler,
                    confidence=min(start.confidence, end.confidence),
                    influence=entry.influence,
                )
                channel = target_pose.channel_for(semantic_id)
                channel.rest_direction = target_direction.copy()
                channel.desired_direction = direction.copy()
                parent_delta = (
                    (channel.desired_rotation @ rest_rotation.inverted_safe())
                    .to_quaternion()
                    .normalized()
                )
                if clamped:
                    warning = (
                        f"{semantic_id.replace('_', ' ').title()} articulation "
                        "was limited"
                    )
                    if warning not in target_pose.warnings:
                        target_pose.warnings.append(warning)
                changed.add(bone)
    return changed


def finger_world_position_errors(
    armature,
    entries,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Return normalized evaluated endpoint errors for mapped phalanges.

    Rotation values are intentionally ignored: generated rigs can assign
    different local axes and roll to equivalent controls. Each error compares
    the evaluated bone tail with the detected direction projected from its
    evaluated head using that target segment's own length.
    """
    source_basis = source_root_basis(frame)
    rig_basis = target_basis(armature, entries)
    errors = {}
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name, {})
        detected_palm_basis = _detected_palm_basis(hand)
        target_primary, target_lateral = _target_palm_vectors(
            armature,
            entries,
            side_suffix,
        )
        target_palm_basis = _orthonormal_palm_basis(
            target_primary,
            target_lateral,
        )
        hand_bone = evaluated_semantic_bone(
            armature,
            entries,
            f"HAND_{side_suffix}",
        )
        hand_delta = None
        if hand_bone is not None:
            hand_delta = (
                (
                    hand_bone.matrix.to_3x3()
                    @ hand_bone.bone.matrix_local.to_3x3().inverted_safe()
                )
                .to_quaternion()
                .normalized()
            )
        for role_base, pair in HAND_JOINT_PAIRS.items():
            semantic_id = f"{role_base}_{side_suffix}"
            bone = evaluated_semantic_bone(armature, entries, semantic_id)
            start, end = hand.get(pair[0]), hand.get(pair[1])
            if (
                bone is None
                or start is None
                or end is None
                or not start.valid
                or not end.valid
            ):
                continue
            raw_direction = Vector(end.position) - Vector(start.position)
            if (
                detected_palm_basis is not None
                and target_palm_basis is not None
                and hand_delta is not None
            ):
                palm_local = detected_palm_basis.inverted_safe() @ raw_direction
                direction = hand_delta @ (target_palm_basis @ palm_local)
            else:
                direction = _to_target_direction(
                    raw_direction,
                    source_basis,
                    rig_basis,
                )
            if direction.length < 1.0e-8:
                continue
            segment_length = max(float(bone.bone.length), 1.0e-8)
            expected_tail = bone.head + direction.normalized() * segment_length
            errors[semantic_id] = (bone.tail - expected_tail).length / segment_length
    return errors


def finger_palm_roll_errors(
    armature,
    entries,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Return axial-roll error relative to parent-transported rest roll.

    Joint positions constrain each phalanx direction but not its roll. This
    diagnostic verifies that the missing degree of freedom follows the solved
    wrist and preceding phalanx while articulation contributes only swing.
    Values are radians in the shortest signed range.
    """
    errors = {}
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name, {})
        hand_bone = evaluated_semantic_bone(armature, entries, f"HAND_{side_suffix}")
        if not hand or hand_bone is None:
            continue
        hand_rest_rotation = hand_bone.bone.matrix_local.to_3x3()
        palm_delta = (
            (hand_bone.matrix.to_3x3() @ hand_rest_rotation.inverted_safe())
            .to_quaternion()
            .normalized()
        )
        parent_deltas = {
            finger: palm_delta
            for finger in ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
        }
        palm_primary, palm_lateral = _target_palm_vectors(
            armature, entries, side_suffix
        )
        palm_normal = (
            palm_primary.cross(palm_lateral).normalized()
            if palm_primary is not None
            and palm_lateral is not None
            and palm_primary.cross(palm_lateral).length > 1.0e-8
            else None
        )
        for role_base in HAND_JOINT_PAIRS:
            semantic_id = f"{role_base}_{side_suffix}"
            bone = evaluated_semantic_bone(armature, entries, semantic_id)
            if bone is None:
                continue
            direction = bone.tail - bone.head
            if direction.length < 1.0e-8:
                continue
            direction.normalize()
            rest_rotation = bone.bone.matrix_local.to_3x3()
            pose_rotation = bone.matrix.to_3x3()
            rest_direction = bone.bone.tail_local - bone.bone.head_local
            if rest_direction.length < 1.0e-8:
                continue
            finger, segment_text = role_base.rsplit("_", 1)
            segment = int(segment_text)
            limits = _FINGER_ARTICULATION_LIMITS[
                "THUMB" if finger == "THUMB" else "FINGER"
            ]
            expected_delta, _clamped = _bounded_parent_relative_direction_delta(
                rest_direction,
                direction,
                parent_deltas[finger],
                limits[segment - 1],
                bend_reference=(
                    palm_normal if finger != "THUMB" and segment > 1 else None
                ),
            )
            parent_deltas[finger] = expected_delta
            actual_delta = (
                (pose_rotation @ rest_rotation.inverted_safe())
                .to_quaternion()
                .normalized()
            )
            residual = actual_delta @ expected_delta.inverted()
            errors[semantic_id] = _signed_twist_angle(residual, direction)
    return errors


def palm_lateral_errors(
    armature,
    entries,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Return evaluated index-to-pinky palm-axis errors in radians.

    Unlike a finger endpoint check, this observes the signed palm frame and
    therefore exposes a whole-hand 180-degree roll even when every joint
    position is otherwise exact.
    """
    errors: dict[str, float] = {}
    source_basis = source_root_basis(frame)
    rig_basis = target_basis(armature, entries)
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name)
        if not hand:
            continue
        index = hand.get("INDEX_FINGER_MCP")
        pinky = hand.get("PINKY_MCP")
        if not index or not pinky or not index.valid or not pinky.valid:
            continue
        rest_primary, rest_lateral = _target_palm_vectors(
            armature, entries, side_suffix
        )
        bone = evaluated_semantic_bone(armature, entries, f"HAND_{side_suffix}")
        if (
            bone is None
            or rest_primary is None
            or rest_lateral is None
            or rest_primary.length < 1.0e-8
            or rest_lateral.length < 1.0e-8
        ):
            continue
        desired_lateral = _to_target_direction(
            Vector(index.position) - Vector(pinky.position),
            source_basis,
            rig_basis,
        )
        rest_rotation = bone.bone.matrix_local.to_3x3()
        delta = (
            (bone.matrix.to_3x3() @ rest_rotation.inverted_safe())
            .to_quaternion()
            .normalized()
        )
        observed_lateral = delta @ rest_lateral
        desired_primary = delta @ rest_primary
        observed_lateral -= (
            desired_primary
            * observed_lateral.dot(desired_primary)
            / max(desired_primary.length_squared, 1.0e-12)
        )
        desired_lateral -= (
            desired_primary
            * desired_lateral.dot(desired_primary)
            / max(desired_primary.length_squared, 1.0e-12)
        )
        if observed_lateral.length < 1.0e-8 or desired_lateral.length < 1.0e-8:
            continue
        dot = min(
            max(
                observed_lateral.normalized().dot(desired_lateral.normalized()),
                -1.0,
            ),
            1.0,
        )
        errors[side_name] = math.acos(dot)
    return errors


def _finite_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _expression_value(entry, raw_value: float) -> float:
    value = (_finite_float(raw_value) - entry.offset) * entry.gain
    if entry.invert:
        value = entry.clamp_max - value
    value *= entry.influence
    return min(max(value, entry.clamp_min), entry.clamp_max)


def facs_values(blendshapes: dict[str, float]) -> dict[str, float]:
    """Approximate FACS AU intensities from MediaPipe blendshape coefficients.

    MediaPipe Face Landmarker emits ARKit-style expression coefficients rather
    than FACS action units. Keep this provider-to-standard conversion separate
    from destination mapping so another detector can supply native AU values
    later without changing the rig adapters.
    """

    def get(name: str) -> float:
        return min(
            max(_finite_float(blendshapes.get(name, 0.0)), 0.0),
            1.0,
        )

    bilateral_sources = {
        "AU01": ("browInnerUp", "browInnerUp"),
        "AU02": ("browOuterUpLeft", "browOuterUpRight"),
        "AU04": ("browDownLeft", "browDownRight"),
        "AU05": ("eyeWideLeft", "eyeWideRight"),
        "AU06": ("cheekSquintLeft", "cheekSquintRight"),
        "AU07": ("eyeSquintLeft", "eyeSquintRight"),
        "AU09": ("noseSneerLeft", "noseSneerRight"),
        "AU10": ("mouthUpperUpLeft", "mouthUpperUpRight"),
        "AU12": ("mouthSmileLeft", "mouthSmileRight"),
        "AU14": ("mouthDimpleLeft", "mouthDimpleRight"),
        "AU15": ("mouthFrownLeft", "mouthFrownRight"),
        "AU16": ("mouthLowerDownLeft", "mouthLowerDownRight"),
        "AU20": ("mouthStretchLeft", "mouthStretchRight"),
        "AU23": ("mouthPressLeft", "mouthPressRight"),
        "AU45": ("eyeBlinkLeft", "eyeBlinkRight"),
    }
    values: dict[str, float] = {}
    for action_unit, (left_source, right_source) in bilateral_sources.items():
        left = get(left_source)
        right = get(right_source)
        values[action_unit] = max(left, right)
        values[f"{action_unit}_L"] = left
        values[f"{action_unit}_R"] = right

    left_press = get("mouthPressLeft")
    right_press = get("mouthPressRight")
    jaw_open = get("jawOpen")
    values.update(
        {
            "AU17": get("mouthShrugLower"),
            "AU18": get("mouthPucker"),
            "AU22": get("mouthFunnel"),
            "AU24": max(left_press, right_press),
            "AU25": jaw_open,
            "AU26": jaw_open,
            "AU27": max(
                jaw_open,
                get("mouthStretchLeft"),
                get("mouthStretchRight"),
            ),
            "AU28": max(get("mouthRollUpper"), get("mouthRollLower")),
        }
    )
    return values


def _matching_shape_blocks(armature, entry):
    yield from mapping.matching_shape_blocks(armature, entry)


def _apply_shapes(armature, entries, frame, scheme: str, target_pose: TargetPose):
    changed = []
    expression_values = (
        facs_values(frame.blendshapes) if scheme == "FACS" else frame.blendshapes
    )
    for name, raw_value in expression_values.items():
        entry = entries.get(f"{scheme}_{name}")
        if not entry or not entry.enabled or not entry.target_name:
            continue
        value = _expression_value(entry, raw_value)
        for obj, block in _matching_shape_blocks(armature, entry):
            value_clamped = min(max(value, block.slider_min), block.slider_max)
            target_pose.set_shape(obj, block, value_clamped)
            changed.append((obj, block))
    return changed


def _axis(axis_name: str) -> Vector:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    return {
        "X": Vector((sign, 0.0, 0.0)),
        "Y": Vector((0.0, sign, 0.0)),
        "Z": Vector((0.0, 0.0, sign)),
    }[axis_name[-1]]


def _secondary_axis(primary: Vector, preferred: Vector) -> Vector:
    """Return a stable perpendicular control axis for two-axis gaze."""
    secondary = preferred - primary * preferred.dot(primary)
    if secondary.length < 1.0e-8:
        fallback = Vector((1.0, 0.0, 0.0))
        secondary = fallback - primary * fallback.dot(primary)
    return secondary.normalized()


def _signed_face_value(entry, value: float) -> float:
    """Apply mapping calibration without unsigned expression clamping."""
    result = (_finite_float(value) - entry.offset) * entry.gain
    if entry.invert:
        result = -result
    return result * entry.influence


def _face_control_values(blendshapes: dict[str, float]) -> dict[str, float]:
    def get(name: str) -> float:
        return _finite_float(blendshapes.get(name, 0.0))

    return {
        "FACE_JAW_OPEN": get("jawOpen"),
        "FACE_JAW_SIDE": get("jawRight") - get("jawLeft"),
        "FACE_BROW_INNER": get("browInnerUp"),
        "FACE_BROW_OUTER_L": get("browOuterUpLeft") - get("browDownLeft"),
        "FACE_BROW_OUTER_R": get("browOuterUpRight") - get("browDownRight"),
        "FACE_LID_L": get("eyeBlinkLeft"),
        "FACE_LID_R": get("eyeBlinkRight"),
        "FACE_CHEEK_L": max(get("cheekSquintLeft"), get("cheekPuff")),
        "FACE_CHEEK_R": max(get("cheekSquintRight"), get("cheekPuff")),
        "FACE_MOUTH_CORNER_L": get("mouthSmileLeft") - get("mouthFrownLeft"),
        "FACE_MOUTH_CORNER_R": get("mouthSmileRight") - get("mouthFrownRight"),
        "FACE_LIP_TOP": max(
            get("mouthUpperUpLeft"), get("mouthUpperUpRight"), get("mouthShrugUpper")
        ),
        "FACE_LIP_BOTTOM": max(
            get("mouthLowerDownLeft"),
            get("mouthLowerDownRight"),
            get("mouthShrugLower"),
        ),
    }


def _eye_gaze(blendshapes: dict[str, float], side: str) -> tuple[float, float]:
    def get(name: str) -> float:
        return _finite_float(blendshapes.get(name, 0.0))

    if side == "L":
        horizontal = get("eyeLookOutLeft") - get("eyeLookInLeft")
        vertical = get("eyeLookUpLeft") - get("eyeLookDownLeft")
    else:
        horizontal = get("eyeLookInRight") - get("eyeLookOutRight")
        vertical = get("eyeLookUpRight") - get("eyeLookDownRight")
    return horizontal, vertical


def _apply_eye_gaze(
    armature,
    entries,
    frame,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    changed = set()
    for semantic_id, side in (("EYE_L", "L"), ("EYE_R", "R")):
        entry = entries.get(semantic_id)
        bone = _mapped_bone(armature, entries, semantic_id)
        if not entry or not bone:
            continue
        horizontal, vertical = _eye_gaze(frame.blendshapes, side)
        if entry.channel_mode == "LOCATION":
            scale = max(bone.length * 0.3, 0.01)
            horizontal_axis = _axis(entry.axis)
            vertical_axis = _secondary_axis(horizontal_axis, Vector((0.0, 0.0, 1.0)))
            target_pose.set_location(
                semantic_id,
                bone,
                (
                    horizontal_axis * _signed_face_value(entry, horizontal)
                    + vertical_axis * _signed_face_value(entry, vertical)
                )
                * scale,
            )
        else:
            vertical_axis = _axis(entry.axis)
            horizontal_axis = _secondary_axis(vertical_axis, Vector((0.0, 0.0, 1.0)))
            rotation = Quaternion(
                horizontal_axis,
                _signed_face_value(entry, horizontal) * 0.35,
            )
            rotation @= Quaternion(
                vertical_axis,
                -_signed_face_value(entry, vertical) * 0.35,
            )
            target_pose.set_basis_rotation(
                semantic_id,
                bone,
                rotation,
                correction=entry.correction_euler,
            )
        changed.add(bone)
    return changed


def _apply_face_controls(
    armature,
    entries,
    frame,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    changed = set()
    values = _face_control_values(frame.blendshapes)
    rotation_groups = {}
    location_groups = {}
    for semantic_id, value in values.items():
        entry = entries.get(semantic_id)
        bone = _mapped_bone(armature, entries, semantic_id)
        if not entry or not bone:
            continue
        evaluated = _expression_value(entry, value)
        if entry.channel_mode == "CUSTOM_PROPERTY" and entry.property_name:
            target_pose.set_property(
                semantic_id,
                bone,
                entry.property_name,
                evaluated,
            )
            changed.add(bone)
        elif entry.channel_mode == "LOCATION":
            location_groups.setdefault(bone, []).append((entry, evaluated))
        else:
            rotation_groups.setdefault(bone, []).append((entry, evaluated))

    for bone, controls in rotation_groups.items():
        rotation = Quaternion()
        correction = Vector((0.0, 0.0, 0.0))
        for entry, value in controls:
            rotation @= Quaternion(_axis(entry.axis), value * math.radians(25.0))
            correction += Vector(entry.correction_euler)
        target_pose.set_basis_rotation(
            controls[0][0].semantic_id,
            bone,
            rotation,
            correction=correction,
        )
        changed.add(bone)

    for bone, controls in location_groups.items():
        direction = Vector((0.0, 0.0, 0.0))
        for entry, value in controls:
            direction += _axis(entry.axis) * value
        target_pose.set_location(
            controls[0][0].semantic_id,
            bone,
            direction * max(bone.length * 0.2, 0.005),
        )
        changed.add(bone)

    return changed


def _rotation_data_path(pose_bone) -> str:
    if pose_bone.rotation_mode == "QUATERNION":
        return "rotation_quaternion"
    if pose_bone.rotation_mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    return "rotation_euler"


def _action_fcurves(action) -> Iterator[object]:
    """Yield F-Curves from a Blender 5.2 layered Action."""
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                yield from channelbag.fcurves


def _set_key_interpolation(
    datablock,
    data_path: str,
    frame: float,
    interpolation: str,
) -> None:
    animation_data = getattr(datablock, "animation_data", None)
    action = animation_data.action if animation_data else None
    if not action:
        return
    for fcurve in _action_fcurves(action):
        if fcurve.data_path != data_path:
            continue
        for point in fcurve.keyframe_points:
            if abs(float(point.co.x) - float(frame)) < 1.0e-5:
                point.interpolation = interpolation


def _previous_key_values(
    pose_bone,
    data_path: str,
    frame: float,
    size: int,
) -> list[float] | None:
    animation = pose_bone.id_data.animation_data
    action = animation.action if animation else None
    if action is None:
        return None
    resolved_path = pose_bone.path_from_id(data_path)
    curves = {
        curve.array_index: curve
        for curve in _action_fcurves(action)
        if curve.data_path == resolved_path
    }
    if any(index not in curves for index in range(size)):
        return None
    previous_frames = [
        float(point.co.x)
        for curve in curves.values()
        for point in curve.keyframe_points
        if float(point.co.x) < float(frame) - 1.0e-5
    ]
    if not previous_frames:
        return None
    previous_frame = max(previous_frames)
    return [float(curves[index].evaluate(previous_frame)) for index in range(size)]


def _make_rotation_key_compatible(
    pose_bone,
    data_path: str,
    frame: float,
) -> None:
    """Choose the closest equivalent rotation to the preceding key."""
    if data_path == "rotation_quaternion":
        previous = _previous_key_values(pose_bone, data_path, frame, 4)
        if previous is None:
            return
        current = pose_bone.rotation_quaternion.copy()
        prior = Quaternion(previous)
        if current.dot(prior) < 0.0:
            current.negate()
            pose_bone.rotation_quaternion = current
        return
    if data_path == "rotation_axis_angle":
        previous = _previous_key_values(pose_bone, data_path, frame, 4)
        if previous is None:
            return
        current = pose_bone.rotation_axis_angle
        axis = Vector(current[1:4])
        if axis.length < 1.0e-8:
            axis = Vector(previous[1:4])
        if axis.length < 1.0e-8:
            axis = Vector((0.0, 1.0, 0.0))
        axis.normalize()
        previous_axis = Vector(previous[1:4])
        angle = float(current[0])
        if previous_axis.length > 1.0e-8 and axis.dot(previous_axis.normalized()) < 0.0:
            axis.negate()
            angle = -angle
        while angle - previous[0] > math.pi:
            angle -= math.tau
        while angle - previous[0] < -math.pi:
            angle += math.tau
        pose_bone.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        return
    previous = _previous_key_values(pose_bone, data_path, frame, 3)
    if previous is None:
        return
    current = pose_bone.rotation_euler.copy()
    current.make_compatible(Euler(previous, pose_bone.rotation_mode))
    pose_bone.rotation_euler = current


def _rotation_values_compatible_with(
    pose_bone,
    data_path: str,
    previous: Sequence[float] | None,
) -> list[float]:
    """Make a rotation continuous using an in-memory preceding sample."""
    if data_path == "rotation_quaternion":
        current = pose_bone.rotation_quaternion.copy()
        if previous is not None and current.dot(Quaternion(previous)) < 0.0:
            current.negate()
            pose_bone.rotation_quaternion = current
        return [float(value) for value in current]
    if data_path == "rotation_axis_angle":
        current = pose_bone.rotation_axis_angle
        axis = Vector(current[1:4])
        if axis.length < 1.0e-8 and previous is not None:
            axis = Vector(previous[1:4])
        if axis.length < 1.0e-8:
            axis = Vector((0.0, 1.0, 0.0))
        axis.normalize()
        angle = float(current[0])
        if previous is not None:
            previous_axis = Vector(previous[1:4])
            if (
                previous_axis.length > 1.0e-8
                and axis.dot(previous_axis.normalized()) < 0.0
            ):
                axis.negate()
                angle = -angle
            while angle - previous[0] > math.pi:
                angle -= math.tau
            while angle - previous[0] < -math.pi:
                angle += math.tau
        values = [angle, axis.x, axis.y, axis.z]
        pose_bone.rotation_axis_angle = values
        return values
    current = pose_bone.rotation_euler.copy()
    if previous is not None:
        current.make_compatible(Euler(previous, pose_bone.rotation_mode))
        pose_bone.rotation_euler = current
    return [float(value) for value in current]


def _key_bone_from_sample(
    pose_bone,
    frame: float,
    *,
    location: bool,
    rotation: bool,
    previous_rotations: dict[tuple[str, str], list[float]],
) -> None:
    """Insert one pre-solved sample without rescanning the growing action."""
    if location:
        pose_bone.keyframe_insert(
            data_path="location",
            frame=frame,
            group=pose_bone.name,
        )
    if not rotation:
        return
    data_path = _rotation_data_path(pose_bone)
    key = (pose_bone.name, data_path)
    previous = previous_rotations.get(key)
    if previous is None:
        size = 4 if data_path != "rotation_euler" else 3
        previous = _previous_key_values(
            pose_bone,
            data_path,
            frame,
            size,
        )
    previous_rotations[key] = _rotation_values_compatible_with(
        pose_bone,
        data_path,
        previous,
    )
    pose_bone.keyframe_insert(
        data_path=data_path,
        frame=frame,
        group=pose_bone.name,
    )


def _key_bone(
    pose_bone,
    frame: float,
    *,
    location: bool = True,
    rotation: bool = True,
) -> None:
    if location:
        pose_bone.keyframe_insert(
            data_path="location", frame=frame, group=pose_bone.name
        )
        _set_key_interpolation(
            pose_bone.id_data,
            pose_bone.path_from_id("location"),
            frame,
            "LINEAR",
        )
    if rotation:
        rotation_path = _rotation_data_path(pose_bone)
        _make_rotation_key_compatible(pose_bone, rotation_path, frame)
        pose_bone.keyframe_insert(
            data_path=rotation_path,
            frame=frame,
            group=pose_bone.name,
        )
        _set_key_interpolation(
            pose_bone.id_data,
            pose_bone.path_from_id(rotation_path),
            frame,
            "LINEAR",
        )


def _key_object_transform(obj, frame: float) -> None:
    obj.keyframe_insert(data_path="location", frame=frame)
    rotation_path = (
        "rotation_quaternion"
        if obj.rotation_mode == "QUATERNION"
        else "rotation_axis_angle"
        if obj.rotation_mode == "AXIS_ANGLE"
        else "rotation_euler"
    )
    animation = obj.animation_data
    action = animation.action if animation else None
    if action is not None:
        size = 4 if rotation_path != "rotation_euler" else 3
        curves = {
            curve.array_index: curve
            for curve in _action_fcurves(action)
            if curve.data_path == rotation_path
        }
        previous_frames = [
            float(point.co.x)
            for curve in curves.values()
            for point in curve.keyframe_points
            if float(point.co.x) < float(frame) - 1.0e-5
        ]
        if previous_frames and all(index in curves for index in range(size)):
            previous_frame = max(previous_frames)
            previous = [
                float(curves[index].evaluate(previous_frame)) for index in range(size)
            ]
            if rotation_path == "rotation_quaternion":
                current = obj.rotation_quaternion.copy()
                if current.dot(Quaternion(previous)) < 0.0:
                    current.negate()
                    obj.rotation_quaternion = current
            elif rotation_path == "rotation_euler":
                current = obj.rotation_euler.copy()
                current.make_compatible(Euler(previous, obj.rotation_mode))
                obj.rotation_euler = current
            else:
                current = obj.rotation_axis_angle
                axis = Vector(current[1:4])
                if axis.length < 1.0e-8:
                    axis = Vector(previous[1:4])
                if axis.length < 1.0e-8:
                    axis = Vector((0.0, 0.0, 1.0))
                axis.normalize()
                previous_axis = Vector(previous[1:4])
                angle = float(current[0])
                if (
                    previous_axis.length > 1.0e-8
                    and axis.dot(previous_axis.normalized()) < 0.0
                ):
                    axis.negate()
                    angle = -angle
                while angle - previous[0] > math.pi:
                    angle -= math.tau
                while angle - previous[0] < -math.pi:
                    angle += math.tau
                obj.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
    obj.keyframe_insert(data_path=rotation_path, frame=frame)
    _set_key_interpolation(obj, "location", frame, "LINEAR")
    _set_key_interpolation(obj, rotation_path, frame, "LINEAR")


def _remove_keys_in_interval(
    action,
    data_paths: set[str],
    start_frame: float,
    end_frame: float,
) -> None:
    """Remove keys for the requested channels within a replacement interval."""
    for curve in _action_fcurves(action):
        if curve.data_path not in data_paths:
            continue
        for point in reversed(list(curve.keyframe_points)):
            if start_frame - 1.0e-5 <= float(point.co.x) <= end_frame + 1.0e-5:
                curve.keyframe_points.remove(point, fast=True)
        curve.update()


def _key_shape(
    block,
    frame: float,
    *,
    set_interpolation: bool = True,
) -> None:
    block.keyframe_insert(data_path="value", frame=frame)
    if set_interpolation:
        _set_key_interpolation(
            block.id_data,
            block.path_from_id("value"),
            frame,
            "LINEAR",
        )


def _ensure_action(armature) -> bpy.types.Action:
    animation = armature.animation_data_create()
    if animation.action is None:
        animation.action = bpy.data.actions.new(f"{armature.name}_MediaToPose")
    return animation.action


def prepare_limb_ownership(armature) -> None:
    """Populate metadata without changing or keying FK/IK switches."""
    mapping.configure_fk_switches(
        armature,
        mapping.profile_identifier(armature.mtp_rig_config),
    )


def limb_ownership(armature) -> dict[tuple[str, str], dict]:
    """Snapshot current per-limb FK/IK ownership without changing the rig."""
    config = armature.mtp_rig_config
    result = {}
    for chain_name, chain_id in (("arm", "ARM"), ("leg", "LEG")):
        for side_name, side in (("l", "L"), ("r", "R")):
            switch_name = getattr(config, f"{chain_name}_fk_switch_bone_{side_name}")
            property_name = getattr(
                config, f"{chain_name}_fk_switch_property_{side_name}"
            )
            switch = armature.pose.bones.get(switch_name)
            fk_value = float(
                getattr(
                    config,
                    f"{chain_name}_fk_switch_value_{side_name}",
                )
            )
            ik_value = float(
                getattr(
                    config,
                    f"{chain_name}_ik_switch_value_{side_name}",
                )
            )
            current = (
                float(switch[property_name])
                if switch and property_name and property_name in switch
                else fk_value
            )
            mode = "IK" if abs(current - ik_value) < abs(current - fk_value) else "FK"
            result[(chain_id, side)] = {
                "mode": mode,
                "switch": switch,
                "property": property_name,
                "value": current,
                "target": armature.pose.bones.get(
                    getattr(
                        config,
                        f"{chain_name}_ik_target_bone_{side_name}",
                    )
                ),
                "pole": armature.pose.bones.get(
                    getattr(
                        config,
                        f"{chain_name}_ik_pole_bone_{side_name}",
                    )
                ),
                "toe": (
                    armature.pose.bones.get(
                        getattr(
                            config,
                            f"leg_ik_toe_bone_{side_name}",
                        )
                    )
                    if chain_name == "leg"
                    else None
                ),
                "rotation_bend": (
                    armature.pose.bones.get(
                        ("upper_arm_ik." if chain_name == "arm" else "thigh_ik.") + side
                    )
                    if (
                        switch
                        and "pole_vector" in switch
                        and not bool(switch["pole_vector"])
                    )
                    else None
                ),
            }
    return result


def validate_limb_switch_interval(
    armature,
    start: float,
    end: float,
) -> str | None:
    """Reject a bake interval containing an existing ownership transition."""
    if end <= start:
        return None
    animation = armature.animation_data
    action = animation.action if animation else None
    if not action:
        return None
    for info in limb_ownership(armature).values():
        switch = info["switch"]
        property_name = info["property"]
        if not switch or not property_name:
            continue
        data_path = switch.path_from_id(f'["{property_name}"]')
        for curve in _action_fcurves(action):
            if curve.data_path != data_path:
                continue
            start_value = float(curve.evaluate(start))
            if abs(float(curve.evaluate(end)) - start_value) > 1.0e-5:
                return (
                    f"{switch.name} changes FK/IK ownership inside the "
                    "bake range; bake mode-consistent segments"
                )
            for point in curve.keyframe_points:
                frame = float(point.co.x)
                if (
                    start < frame <= end
                    and abs(float(curve.evaluate(frame)) - start_value) > 1.0e-5
                ):
                    return (
                        f"{switch.name} changes FK/IK ownership inside the "
                        "bake range; bake mode-consistent segments"
                    )
            # Bezier handles can overshoot or cross ownership thresholds
            # between key coordinates. Sample subframes across the interval,
            # including the actual integer frames a bake normally writes.
            sample = math.floor(start * 4.0 + 1.0) / 4.0
            while sample < end:
                if abs(float(curve.evaluate(sample)) - start_value) > 1.0e-5:
                    return (
                        f"{switch.name} changes FK/IK ownership inside the "
                        "bake range; bake mode-consistent segments"
                    )
                sample += 0.25
    return None


def _ik_pole_location(
    upper_matrix: Matrix,
    lower_matrix: Matrix,
    end_matrix: Matrix,
    pole,
    rest_geometry=None,
) -> Vector:
    root = upper_matrix.translation
    joint = lower_matrix.translation
    end = end_matrix.translation
    line = end - root
    rest_root = rest_joint = rest_end = None
    if rest_geometry is not None:
        rest_root, rest_joint, rest_end = rest_geometry
    if line.length < 1.0e-8:
        if rest_joint is not None:
            return joint + (pole.bone.matrix_local.translation - rest_joint)
        return joint.copy()
    axis = line.normalized()
    bend = joint - (root + axis * (joint - root).dot(axis))
    expected_bend = None
    if rest_root is not None and rest_joint is not None and rest_end is not None:
        rest_line = rest_end - rest_root
        if rest_line.length > 1.0e-8:
            rest_axis = rest_line.normalized()
            rest_bend = rest_joint - (
                rest_root + rest_axis * (rest_joint - rest_root).dot(rest_axis)
            )
            if rest_bend.length < 1.0e-8:
                rest_reference = pole.bone.matrix_local.translation - rest_joint
                rest_bend = rest_reference - (rest_axis * rest_reference.dot(rest_axis))
            if rest_bend.length > 1.0e-8:
                if rest_axis.dot(axis) < -0.9999:
                    transport = Quaternion(rest_bend.normalized(), math.pi)
                else:
                    transport = rest_axis.rotation_difference(axis)
                expected_bend = transport @ rest_bend
    straight_threshold = max(line.length * 1.0e-4, 1.0e-6)
    if bend.length < straight_threshold and expected_bend is not None:
        bend = expected_bend
    elif (
        expected_bend is not None
        and bend.length > 1.0e-8
        and bend.dot(expected_bend) < 0.0
    ):
        # Elbows and knees do not acquire a reverse bend merely because a
        # noisy observation crosses the straight-line singularity.
        bend.negate()
    if bend.length < 1.0e-8:
        bend = axis.cross(Vector((0.0, 0.0, 1.0)))
        if bend.length < 1.0e-8:
            bend = axis.cross(Vector((0.0, 1.0, 0.0)))
    rest_distance = 0.0
    if rest_joint is not None:
        rest_distance = (pole.bone.matrix_local.translation - rest_joint).length
    distance = max(
        rest_distance,
        line.length * 0.5,
        0.01,
    )
    return joint + bend.normalized() * distance


def _ik_rest_geometry(upper, lower):
    return (
        upper.bone.head_local.copy(),
        lower.bone.head_local.copy(),
        lower.bone.tail_local.copy(),
    )


def _ik_rest_pole_distance(pole, rest_geometry, chain_length):
    _rest_root, rest_joint, _rest_end = rest_geometry
    return max(
        (pole.bone.matrix_local.translation - rest_joint).length,
        float(chain_length) * 0.5,
        0.01,
    )


def _ik_channel_direction(
    target_pose: TargetPose,
    channel,
    pose_matrix: Matrix,
) -> Vector:
    """Return the anatomical direction represented by a generated control."""
    if (
        channel.desired_direction is not None
        and channel.desired_direction.length > 1.0e-8
    ):
        return channel.desired_direction.normalized()
    rest_rotation = channel.bone.bone.matrix_local.to_quaternion()
    delta = pose_matrix.to_quaternion() @ rest_rotation.inverted()
    rest_direction = (
        channel.rest_direction
        if channel.rest_direction is not None
        else _target_body_direction(
            target_pose.armature,
            target_pose.entries,
            channel.semantic_id,
        )
    )
    return (delta @ rest_direction).normalized()


def _rest_relative_effector_rotation(
    rest_lower: Quaternion,
    desired_lower: Quaternion,
    rest_effector: Quaternion,
) -> Quaternion:
    """Carry an undetected effector using a deterministic rest relationship."""
    relative = rest_lower.normalized().inverted() @ rest_effector.normalized()
    return (desired_lower.normalized() @ relative).normalized()


def _prepare_ik_output(target_pose: TargetPose) -> list[dict]:
    """Convert processed virtual FK chains into profile IK proposals."""
    proposals = []
    ownership = limb_ownership(target_pose.armature)
    semantics = {
        ("ARM", "L"): ("UPPER_ARM_L", "LOWER_ARM_L", "HAND_L"),
        ("ARM", "R"): ("UPPER_ARM_R", "LOWER_ARM_R", "HAND_R"),
        ("LEG", "L"): ("UPPER_LEG_L", "LOWER_LEG_L", "FOOT_L"),
        ("LEG", "R"): ("UPPER_LEG_R", "LOWER_LEG_R", "FOOT_R"),
    }
    target_pose.resolve()
    for key, roles in semantics.items():
        info = ownership[key]
        if info["mode"] != "IK":
            continue
        target = info["target"]
        pole = info["pole"]
        if not target or not pole:
            raise ValueError(
                f"{key[1]} {key[0].lower()} is in IK mode but its "
                "target or pole mapping is missing"
            )
        channels = [target_pose.channel_for(role) for role in roles]
        if (
            key[0] == "ARM"
            and channels[0] is None
            and channels[1] is None
            and channels[2] is not None
        ):
            end_channel = channels[2]
            end_matrix = target_pose._virtual_pose(end_channel.bone).copy()
            rest_offset = (
                end_channel.bone.bone.matrix_local.inverted_safe()
                @ target.bone.matrix_local
            )
            target_matrix = end_matrix @ rest_offset
            target_matrix.translation = target.matrix.translation
            proposals.append(
                {
                    "key": key,
                    "armature": target_pose.armature,
                    "entries": target_pose.entries,
                    "roles": roles,
                    "target": target,
                    "pole": pole,
                    "target_matrix": target_matrix,
                    "orientation_only": True,
                    "confidence": float(end_channel.confidence),
                }
            )
            target_pose.bones.pop(end_channel.bone.name, None)
            continue
        if channels[0] is not None and channels[1] is not None and channels[2] is None:
            upper_matrix = target_pose._virtual_pose(channels[0].bone).copy()
            lower_matrix = target_pose._virtual_pose(channels[1].bone).copy()
            output_upper = evaluated_semantic_bone(
                target_pose.armature,
                target_pose.entries,
                roles[0],
            )
            output_lower = evaluated_semantic_bone(
                target_pose.armature,
                target_pose.entries,
                roles[1],
            )
            if output_upper is None or output_lower is None:
                continue
            upper_direction = _ik_channel_direction(
                target_pose, channels[0], upper_matrix
            )
            lower_direction = _ik_channel_direction(
                target_pose, channels[1], lower_matrix
            )
            desired_root = upper_matrix.translation.copy()
            desired_joint = desired_root + upper_direction * float(
                output_upper.bone.length
            )
            desired_end = desired_joint + lower_direction * float(
                output_lower.bone.length
            )
            # No reliable hand/foot frame was detected. Carry the rest-pose
            # effector/lower-limb relationship with the solved lower limb.
            # Using the current pose here made output depend on the starting
            # pose and compounded changes across repeated applications.
            desired_target_rotation = _rest_relative_effector_rotation(
                channels[1].bone.bone.matrix_local.to_quaternion(),
                lower_matrix.to_quaternion(),
                target.bone.matrix_local.to_quaternion(),
            )
            target_matrix = Matrix.LocRotScale(
                desired_end,
                desired_target_rotation,
                target.matrix.to_scale(),
            )
            upper_points = Matrix.Translation(desired_root)
            lower_points = Matrix.Translation(desired_joint)
            end_points = Matrix.Translation(desired_end)
            rest_geometry = _ik_rest_geometry(
                output_upper,
                output_lower,
            )
            pole_distance = _ik_rest_pole_distance(
                pole,
                rest_geometry,
                float(output_upper.bone.length) + float(output_lower.bone.length),
            )
            proposal = {
                "key": key,
                "armature": target_pose.armature,
                "entries": target_pose.entries,
                "roles": roles,
                "target": target,
                "pole": pole,
                "target_matrix": target_matrix,
                "pole_location": _ik_pole_location(
                    upper_points,
                    lower_points,
                    end_points,
                    pole,
                    rest_geometry,
                ),
                "pole_rest_geometry": rest_geometry,
                "pole_distance": pole_distance,
                "desired_joint": desired_joint,
                "desired_end": desired_end,
                "upper_direction": upper_direction,
                "lower_direction": lower_direction,
                "upper_length": float(output_upper.bone.length),
                "lower_length": float(output_lower.bone.length),
                "confidence": min(
                    float(channel.confidence)
                    for channel in channels[:2]
                    if channel is not None
                ),
            }
            rotation_bend = info.get("rotation_bend")
            if rotation_bend:
                proposal["rotation_bend"] = rotation_bend
            proposals.append(proposal)
            for channel in channels[:2]:
                target_pose.bones.pop(channel.bone.name, None)
            continue
        if any(channel is None for channel in channels):
            continue
        upper_matrix, lower_matrix, end_matrix = (
            target_pose._virtual_pose(channel.bone).copy() for channel in channels
        )
        output_upper = evaluated_semantic_bone(
            target_pose.armature,
            target_pose.entries,
            roles[0],
        )
        output_lower = evaluated_semantic_bone(
            target_pose.armature,
            target_pose.entries,
            roles[1],
        )
        if output_upper is None or output_lower is None:
            continue

        # Generated FK controls are not always a spatially continuous
        # skeleton (their mechanism parents and offsets can move control
        # origins).  Reconstruct the desired joint positions from the solved
        # absolute rotations and the evaluated deform-chain lengths instead
        # of treating FK control locations as provider joints.
        upper_direction = _ik_channel_direction(target_pose, channels[0], upper_matrix)
        lower_direction = _ik_channel_direction(target_pose, channels[1], lower_matrix)
        desired_root = upper_matrix.translation.copy()
        desired_joint = desired_root + upper_direction * float(output_upper.bone.length)
        desired_end = desired_joint + lower_direction * float(output_lower.bone.length)
        upper_points = upper_matrix.copy()
        lower_points = lower_matrix.copy()
        end_points = end_matrix.copy()
        upper_points.translation = desired_root
        lower_points.translation = desired_joint
        end_points.translation = desired_end
        end_bone = channels[-1].bone
        rest_offset = (
            end_bone.bone.matrix_local.inverted_safe() @ target.bone.matrix_local
        )
        target_matrix = end_matrix @ rest_offset
        target_matrix.translation += desired_end - end_matrix.translation
        rest_geometry = _ik_rest_geometry(
            output_upper,
            output_lower,
        )
        pole_distance = _ik_rest_pole_distance(
            pole,
            rest_geometry,
            float(output_upper.bone.length) + float(output_lower.bone.length),
        )
        pole_location = _ik_pole_location(
            upper_points,
            lower_points,
            end_points,
            pole,
            rest_geometry,
        )
        proposal = {
            "key": key,
            "armature": target_pose.armature,
            "entries": target_pose.entries,
            "roles": roles,
            "target": target,
            "pole": pole,
            "target_matrix": target_matrix,
            "pole_location": pole_location,
            "pole_rest_geometry": rest_geometry,
            "pole_distance": pole_distance,
            "desired_joint": desired_joint,
            "desired_end": desired_end,
            "upper_direction": upper_direction,
            "lower_direction": lower_direction,
            "upper_length": float(output_upper.bone.length),
            "lower_length": float(output_lower.bone.length),
            "confidence": min(float(channel.confidence) for channel in channels),
        }
        rotation_bend = info.get("rotation_bend")
        if rotation_bend:
            proposal["rotation_bend"] = rotation_bend
        if key[0] == "LEG" and info.get("toe"):
            toe_channel = target_pose.channel_for(f"TOE_{key[1]}")
            if toe_channel:
                toe_matrix = target_pose._virtual_pose(toe_channel.bone).copy()
                proposal["toe"] = info["toe"]
                proposal["toe_matrix"] = (
                    toe_matrix
                    @ toe_channel.bone.bone.matrix_local.inverted_safe()
                    @ info["toe"].bone.matrix_local
                )
                target_pose.bones.pop(toe_channel.bone.name, None)
        proposals.append(proposal)
        for channel in channels:
            target_pose.bones.pop(channel.bone.name, None)
    target_pose.invalidate()
    return proposals


def _set_pose_translation_preserve_basis_rotation(
    bone,
    target_translation: Vector,
    *,
    attempts: int = 3,
    solver: PoseTranslationSolver | None = None,
) -> None:
    """Place a control in evaluated pose space while preserving rotation."""
    (solver or PoseTranslationSolver()).solve(
        bone,
        target_translation,
        attempts=attempts,
    )


def _commit_ik_output(proposals: list[dict]) -> set[object]:
    changed = set()
    translation_solver = PoseTranslationSolver()
    for proposal in proposals:
        target = proposal["target"]
        if proposal.get("orientation_only"):
            oriented = proposal["target_matrix"].copy()
            oriented.translation = target.matrix.translation
            target.matrix = oriented
            bpy.context.view_layer.update()
            changed.add(target)
            continue
        rotation_bend = proposal.get("rotation_bend")
        root = evaluated_semantic_bone(
            proposal["armature"],
            proposal["entries"],
            proposal["roles"][0],
        )
        if root is not None:
            previous_end = proposal["desired_end"].copy()
            proposal["desired_joint"] = (
                root.head + proposal["upper_direction"] * proposal["upper_length"]
            )
            proposal["desired_end"] = (
                proposal["desired_joint"]
                + proposal["lower_direction"] * proposal["lower_length"]
            )
            proposal["target_matrix"].translation += (
                proposal["desired_end"] - previous_end
            )
            upper_points = Matrix.Translation(root.head)
            lower_points = Matrix.Translation(proposal["desired_joint"])
            end_points = Matrix.Translation(proposal["desired_end"])
            proposal["pole_location"] = _ik_pole_location(
                upper_points,
                lower_points,
                end_points,
                proposal["pole"],
                proposal.get("pole_rest_geometry"),
            )
        # IK ownership is position-first.  Preserve the control's current
        # orientation while placing the effector and solving its bend plane;
        # monocular palm/foot roll must not displace a wrist or ankle.
        position_matrix = target.matrix.copy()
        position_matrix.translation = proposal["target_matrix"].translation
        target.matrix = position_matrix
        bpy.context.view_layer.update()
        pole = proposal["pole"]
        if rotation_bend is None:
            _set_pose_translation_preserve_basis_rotation(
                pole,
                proposal["pole_location"],
                solver=translation_solver,
            )
        lower = evaluated_semantic_bone(
            proposal["armature"],
            proposal["entries"],
            proposal["roles"][1],
        )
        end_output = evaluated_semantic_bone(
            proposal["armature"],
            proposal["entries"],
            proposal["roles"][2],
        )
        if lower and end_output:
            # Generated controls can have Child Of and mechanism-parent
            # offsets. Correct the effector in evaluated pose space instead
            # of assuming that assigning the visible control matrix is exact.
            for _attempt in range(3):
                error = proposal["desired_end"] - end_output.head
                if error.length < 1.0e-5:
                    break
                corrected = target.matrix.copy()
                corrected.translation += error
                target.matrix = corrected
                bpy.context.view_layer.update()
            if root and rotation_bend is not None:
                # With pole_vector disabled, Rigify exposes only local Y on
                # upper_arm_ik/thigh_ik.  That channel is the roll of the bend
                # plane around the root-to-effector line; assigning an
                # arbitrary global matrix is invalid because X/Z are locked.
                for _attempt in range(8):
                    end_error = proposal["desired_end"] - end_output.head
                    if end_error.length >= 1.0e-5:
                        corrected = target.matrix.copy()
                        corrected.translation += end_error
                        target.matrix = corrected
                        bpy.context.view_layer.update()
                    axis = proposal["desired_end"] - root.head
                    if axis.length < 1.0e-8:
                        break
                    axis.normalize()
                    current_bend = root.tail - (
                        root.head + axis * (root.tail - root.head).dot(axis)
                    )
                    desired_bend = proposal["desired_joint"] - (
                        root.head
                        + axis * (proposal["desired_joint"] - root.head).dot(axis)
                    )
                    if current_bend.length < 1.0e-8 or desired_bend.length < 1.0e-8:
                        break
                    current_bend.normalize()
                    desired_bend.normalize()
                    angle = math.atan2(
                        axis.dot(current_bend.cross(desired_bend)),
                        current_bend.dot(desired_bend),
                    )
                    if abs(angle) < math.radians(0.05) and end_error.length < 1.0e-5:
                        break
                    if abs(angle) >= math.radians(0.05):
                        rotation = rotation_bend.rotation_euler.copy()
                        rotation.y += angle
                        rotation_bend.rotation_euler = rotation
                        bpy.context.view_layer.update()
            if root and rotation_bend is None:
                # Alternate absolute effector and pole placement using only
                # detected/rest geometry. Rotating the current pole offset
                # toward the desired bend made this an iterative feedback
                # solver: applying the same frame again produced a new pose.
                for _attempt in range(3):
                    upper_points = Matrix.Translation(root.head)
                    lower_points = Matrix.Translation(proposal["desired_joint"])
                    end_points = Matrix.Translation(proposal["desired_end"])
                    pole_location = _ik_pole_location(
                        upper_points,
                        lower_points,
                        end_points,
                        pole,
                        proposal.get("pole_rest_geometry"),
                    )
                    _set_pose_translation_preserve_basis_rotation(
                        pole,
                        pole_location,
                        solver=translation_solver,
                    )
                    error = proposal["desired_end"] - end_output.head
                    if error.length < 1.0e-5:
                        break
                    corrected = target.matrix.copy()
                    corrected.translation += error
                    target.matrix = corrected
                    bpy.context.view_layer.update()
            # Apply the available virtual hand/foot orientation only
            # after positional IK has converged, then repair any target-pivot
            # offset introduced by that orientation (common on foot rigs).
            oriented = proposal["target_matrix"].copy()
            oriented.translation = target.matrix.translation
            target.matrix = oriented
            bpy.context.view_layer.update()
            # Auto-Rig Pro parents each pole to its IK effector. Rotating the
            # effector changes the pole constraint's local-to-world Jacobian.
            # Reusing the inverse built before that rotation can move the pole
            # to the opposite side of the limb, producing an apparent
            # 180-degree knee or elbow flip that depends on the starting pose.
            translation_solver.invalidate(pole)
            for _attempt in range(3):
                error = proposal["desired_end"] - end_output.head
                if error.length < 1.0e-5:
                    break
                corrected = target.matrix.copy()
                corrected.translation += error
                target.matrix = corrected
                bpy.context.view_layer.update()
            if root and rotation_bend is None:
                # Target orientation can shift controls whose pivot is not at
                # the deform joint. Restore the same absolute bend plane once
                # more, then repair the endpoint without using current state.
                pole_location = _ik_pole_location(
                    Matrix.Translation(root.head),
                    Matrix.Translation(proposal["desired_joint"]),
                    Matrix.Translation(proposal["desired_end"]),
                    pole,
                    proposal.get("pole_rest_geometry"),
                )
                _set_pose_translation_preserve_basis_rotation(
                    pole,
                    pole_location,
                    solver=translation_solver,
                )
                for _attempt in range(2):
                    error = proposal["desired_end"] - end_output.head
                    if error.length < 1.0e-5:
                        break
                    corrected = target.matrix.copy()
                    corrected.translation += error
                    target.matrix = corrected
                    bpy.context.view_layer.update()
        changed.add(target)
        if rotation_bend is None:
            changed.add(pole)
        if rotation_bend is not None:
            changed.add(rotation_bend)
        toe = proposal.get("toe")
        if toe is not None:
            toe.matrix = proposal["toe_matrix"]
            bpy.context.view_layer.update()
            changed.add(toe)
    return changed


def _iter_commit_target_pose(
    target_pose: TargetPose,
    ik_proposals: list[dict],
) -> Iterator[tuple[str, int, int]]:
    """Commit FK/body and IK channels as one cooperative transaction.

    Blender may redraw between dependency-safe batches. Snapshot only channels
    owned by this frame and restore the complete pre-commit state on failure,
    cancellation, or generator close.
    """
    affected_bones = {channel.bone for channel in target_pose.bones.values()}
    for proposal in ik_proposals:
        for key in ("target", "pole", "rotation_bend", "toe"):
            bone = proposal.get(key)
            if bone is not None:
                affected_bones.add(bone)
    bone_basis = {bone: bone.matrix_basis.copy() for bone in affected_bones}
    property_values = [
        (
            channel.bone,
            property_name,
            property_name in channel.bone,
            channel.bone.get(property_name),
        )
        for channel in target_pose.bones.values()
        for property_name in channel.properties
    ]
    shape_values = [
        (shape.block, float(shape.block.value)) for shape in target_pose.shapes.values()
    ]
    try:
        all_channels = target_pose.bones
        finger_prefixes = ("THUMB_", "INDEX_", "MIDDLE_", "RING_", "PINKY_")
        finger_channels = {
            name: channel
            for name, channel in all_channels.items()
            if channel.semantic_id.startswith(finger_prefixes)
        }
        # Generated rigs finalize the hand control through IK after ordinary
        # body channels. Finger controls are descendants of the evaluated
        # deform hand, so resolving them before that IK output bakes the old
        # parent frame into every phalanx as one large side-wide offset.
        if ik_proposals and finger_channels:
            target_pose.bones = {
                name: channel
                for name, channel in all_channels.items()
                if name not in finger_channels
            }
            target_pose.invalidate()
        pose_commit = target_pose.iter_commit()
        while True:
            try:
                current, total = next(pose_commit)
            except StopIteration as stopped:
                changed_bones, changed_shapes = stopped.value
                break
            yield "Rig channels", int(current), int(total)
        for index, proposal in enumerate(ik_proposals, 1):
            changed_bones.update(_commit_ik_output([proposal]))
            yield "IK controls", index, len(ik_proposals)
        if ik_proposals and finger_channels:
            original_shapes = target_pose.shapes
            target_pose.bones = finger_channels
            target_pose.shapes = {}
            target_pose.invalidate()
            finger_commit = target_pose.iter_commit()
            while True:
                try:
                    current, total = next(finger_commit)
                except StopIteration as stopped:
                    deferred_bones, _deferred_shapes = stopped.value
                    break
                yield "Finger controls", int(current), int(total)
            changed_bones.update(deferred_bones)
            target_pose.bones = all_channels
            target_pose.shapes = original_shapes
            target_pose.invalidate()
        return changed_bones, changed_shapes
    except BaseException:
        if "all_channels" in locals():
            target_pose.bones = all_channels
        if "original_shapes" in locals():
            target_pose.shapes = original_shapes
        target_pose.invalidate()
        for bone, basis in bone_basis.items():
            bone.matrix_basis = basis
        for bone, property_name, existed, value in property_values:
            if existed:
                bone[property_name] = value
            elif property_name in bone:
                del bone[property_name]
        for block, value in shape_values:
            block.value = value
        target_pose.armature.update_tag(refresh={"OBJECT"})
        bpy.context.view_layer.update()
        raise


def _commit_target_pose(
    target_pose: TargetPose,
    ik_proposals: list[dict],
) -> tuple[set[object], list[tuple[object, object]]]:
    iterator = _iter_commit_target_pose(target_pose, ik_proposals)
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


@contextmanager
def keyframe_interpolation_defaults(interpolation: str):
    preferences = bpy.context.preferences.edit
    previous = preferences.keyframe_new_interpolation_type
    preferences.keyframe_new_interpolation_type = interpolation
    try:
        yield
    finally:
        preferences.keyframe_new_interpolation_type = previous


@contextmanager
def linear_keyframe_defaults():
    with keyframe_interpolation_defaults("LINEAR"):
        yield


def build_target_pose(
    armature,
    frame: CanonicalFrame,
    settings,
    *,
    post_process: bool = True,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
    postprocess_config=None,
    rig_plan: rig_planning.RigSolvePlan | None = None,
) -> TargetPose:
    """Create a complete target pose without mutating Blender data."""
    config = armature.mtp_rig_config
    if rig_plan is None:
        entries = mapping.mapping_by_id(config)
        rig_plan = rig_planning.compile_rig_solve_plan(
            armature,
            entries,
            lambda: target_rest_basis(armature, entries),
        )
    else:
        if rig_plan.armature_pointer != int(armature.as_pointer()):
            raise ValueError("Rig solve plan belongs to a different armature")
        entries = rig_plan.entries
    source_space = source_root_basis(frame)
    target_space = target_basis(
        armature,
        entries,
        rest_basis=rig_plan.rest_basis,
    )
    target_pose = TargetPose(
        armature,
        frame,
        entries,
        source_space,
        target_space,
        rig_plan=rig_plan,
    )

    if settings.use_body and frame.body:
        _apply_body(
            armature,
            entries,
            frame,
            source_space,
            target_space,
            target_pose,
        )
    if settings.use_hands and frame.hands:
        _apply_hands(
            armature,
            entries,
            frame,
            source_space,
            target_space,
            target_pose,
        )
    if (
        settings.use_face
        and settings.face_output == "BONES"
        and getattr(settings, "provider", "MEDIAPIPE") == "MEDIAPIPE"
        and frame.blendshapes
    ):
        _apply_eye_gaze(armature, entries, frame, target_pose)
    if settings.use_face and getattr(settings, "provider", "MEDIAPIPE") == "MEDIAPIPE":
        if settings.face_output in {"ARKIT", "FACS"}:
            if frame.blendshapes:
                _apply_shapes(
                    armature,
                    entries,
                    frame,
                    settings.face_output,
                    target_pose,
                )
        elif settings.face_output == "BONES" and frame.blendshapes:
            _apply_face_controls(
                armature,
                entries,
                frame,
                target_pose,
            )

    if post_process:
        run_pipeline(
            target_pose,
            postprocess_config or compile_postprocess_config(config),
            session=session,
            module_filter=module_filter,
        )
    return target_pose


def iter_apply_frame(
    armature,
    frame: CanonicalFrame,
    settings,
    *,
    keyframe: bool = False,
    scene_frame: float = 0.0,
    post_process: bool = True,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
    postprocess_config=None,
) -> Iterator[SequenceApplyProgress]:
    """Apply one frame cooperatively so Blender can redraw between stages."""
    config = armature.mtp_rig_config
    request_entries = getattr(settings, "mapping_entries", None)
    entries = request_entries or mapping.mapping_by_id(config)
    rig_plan = None
    if request_entries:
        rig_plan = rig_planning.compile_rig_solve_plan(
            armature,
            request_entries,
            lambda: target_rest_basis(armature, request_entries),
        )
    yield SequenceApplyProgress("Solving target pose", 60, 100)
    if not bool(getattr(settings, "keep_original_root_position", False)):
        _root_kind, root_target = _sequence_root_target(armature, entries)
        _reset_root_location(root_target)
    target_pose = build_target_pose(
        armature,
        frame,
        settings,
        post_process=False,
        rig_plan=rig_plan,
    )
    if post_process:
        compiled_config = (
            postprocess_config
            or getattr(settings, "postprocess_config", None)
            or compile_postprocess_config(config)
        )
        pipeline = iter_pipeline(
            target_pose,
            compiled_config,
            session=session,
            module_filter=module_filter,
        )
        while True:
            try:
                display_name, index, total = next(pipeline)
            except StopIteration:
                break
            current = 65 + int(20 * (index - 1) / max(total, 1))
            yield SequenceApplyProgress(
                f"Post-processing: {display_name}",
                current,
                100,
            )

    yield SequenceApplyProgress("Preparing rig controls", 88, 100)
    ik_proposals = _prepare_ik_output(target_pose)
    yield SequenceApplyProgress("Committing pose", 93, 100)
    commit = _iter_commit_target_pose(target_pose, ik_proposals)
    while True:
        try:
            commit_stage, _commit_current, _commit_total = next(commit)
        except StopIteration as stopped:
            changed_bones, changed_shapes = stopped.value
            break
        commit_ratio = float(_commit_current) / max(int(_commit_total), 1)
        yield SequenceApplyProgress(
            f"Committing pose: {commit_stage}",
            93 + int(3 * commit_ratio),
            100,
        )
    for warning in target_pose.warnings:
        if warning not in frame.warnings:
            frame.warnings.append(warning)

    yield SequenceApplyProgress("Updating scene", 97, 100)
    bpy.context.view_layer.update()
    if keyframe:
        yield SequenceApplyProgress("Writing keyframe", 98, 100)
        for bone in changed_bones:
            _key_bone(bone, scene_frame)
        for channel in target_pose.bones.values():
            bone = channel.bone
            if bone in changed_bones:
                for property_name in channel.properties:
                    bone.keyframe_insert(
                        data_path=f'["{property_name}"]',
                        frame=scene_frame,
                        group=bone.name,
                    )
                    _set_key_interpolation(
                        bone.id_data,
                        bone.path_from_id(f'["{property_name}"]'),
                        scene_frame,
                        "LINEAR",
                    )
        for _obj, block in changed_shapes:
            _key_shape(block, scene_frame)
    yield SequenceApplyProgress("Finalizing", 100, 100)
    return len(changed_bones), len(changed_shapes)


def apply_frame(
    armature,
    frame: CanonicalFrame,
    settings,
    *,
    keyframe: bool = False,
    scene_frame: float = 0.0,
    post_process: bool = True,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
) -> tuple[int, int]:
    iterator = iter_apply_frame(
        armature,
        frame,
        settings,
        keyframe=keyframe,
        scene_frame=scene_frame,
        post_process=post_process,
        session=session,
        module_filter=module_filter,
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


def _scene_set_float(scene, value: float) -> None:
    integer = math.floor(float(value))
    scene.frame_set(integer, subframe=float(value) - integer)


def _shape_blocks_for_config(armature):
    for entry in armature.mtp_rig_config.mappings:
        if entry.section != "SHAPES" or not entry.target_name:
            continue
        yield from mapping.matching_shape_blocks(armature, entry)


def _sequence_state(armature) -> dict:
    properties = {}
    for bone in armature.pose.bones:
        properties[bone.name] = {
            name: value
            for name in bone.keys()  # noqa: SIM118 - Blender IDProperty API
            if isinstance((value := bone.get(name)), (bool, int, float))
        }
    return {
        "frame": (
            int(bpy.context.scene.frame_current),
            float(bpy.context.scene.frame_subframe),
        ),
        "world": armature.matrix_world.copy(),
        "bones": {bone.name: bone.matrix_basis.copy() for bone in armature.pose.bones},
        "properties": properties,
        "shapes": {
            (block.as_pointer(), block.name): (block, float(block.value))
            for _obj, block in _shape_blocks_for_config(armature)
        },
    }


def _restore_sequence_state(armature, state: dict) -> None:
    frame, subframe = state["frame"]
    bpy.context.scene.frame_set(frame, subframe=subframe)
    armature.matrix_world = state["world"]
    for name, basis in state["bones"].items():
        bone = armature.pose.bones.get(name)
        if bone is not None:
            bone.matrix_basis = basis
    for name, values in state["properties"].items():
        bone = armature.pose.bones.get(name)
        if bone is None:
            continue
        for property_name, value in values.items():
            if property_name in bone:
                bone[property_name] = value
    for block, value in state["shapes"].values():
        block.value = value
    armature.update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()


def _sequence_root_target(armature, entries):
    root = _mapped_bone(armature, entries, "ROOT")
    return ("BONE", root) if root is not None else ("OBJECT", armature)


def _reset_root_location(root_target) -> None:
    """Reset the user-facing root location channels without changing rotation."""
    root_target.location = Vector()
    root_target.id_data.update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()


def _source_body_height(frame: CanonicalFrame) -> float:
    points = body_points(frame)

    def midpoint(first: str, second: str) -> Vector | None:
        left, right = points.get(first), points.get(second)
        if not left or not right or not left.valid or not right.valid:
            return None
        return (Vector(left.position) + Vector(right.position)) * 0.5

    hip = midpoint("LEFT_HIP", "RIGHT_HIP")
    shoulder = midpoint("LEFT_SHOULDER", "RIGHT_SHOULDER")
    ankle = midpoint("LEFT_ANKLE", "RIGHT_ANKLE")
    head_points = [
        points.get(name)
        for name in ("NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR")
    ]
    head_points = [point for point in head_points if point and point.valid]
    head = (
        sum((Vector(point.position) for point in head_points), Vector())
        / len(head_points)
        if head_points
        else None
    )
    lengths = []
    if hip is not None and ankle is not None:
        lengths.append((hip - ankle).length)
    if hip is not None and shoulder is not None:
        lengths.append((hip - shoulder).length)
    if shoulder is not None and head is not None:
        lengths.append((shoulder - head).length)
    return sum(lengths) if len(lengths) >= 2 else 0.0


def _target_body_height(armature, entries) -> float:
    chains = []
    for side in ("L", "R"):
        upper = _mapped_bone(armature, entries, f"UPPER_LEG_{side}")
        lower = _mapped_bone(armature, entries, f"LOWER_LEG_{side}")
        if upper and lower:
            chains.append(float(upper.bone.length + lower.bone.length))
    leg = sum(chains) / len(chains) if chains else 0.0
    pelvis = _mapped_bone(armature, entries, "PELVIS")
    head = _mapped_bone(armature, entries, "HEAD")
    torso = (
        (head.bone.tail_local - pelvis.bone.head_local).length
        if pelvis and head
        else 0.0
    )
    return leg + torso


def _root_motion_sample(frame: CanonicalFrame) -> _RootMotionSample:
    return _RootMotionSample(
        timestamp_seconds=float(frame.timestamp_ms) / 1000.0,
        observation=frame.root_motion,
        source_basis=source_root_basis(frame),
        source_height=_source_body_height(frame),
    )


def _root_motion_deltas(
    armature,
    entries,
    samples: Sequence[_RootMotionSample],
    mode: str,
) -> tuple[list[Vector], list[float], str, list[str]]:
    zeros = [Vector() for _sample in samples]
    yaws = [0.0 for _sample in samples]
    if mode != "AUTOMATIC":
        return zeros, yaws, mode, []
    observations = [
        index
        for index, sample in enumerate(samples)
        if sample.observation is not None and sample.observation.valid
    ]
    if not observations:
        warning = (
            "Automatic root travel is unavailable for this provider/cache; "
            "baked in place"
        )
        return zeros, yaws, "IN_PLACE", [warning]

    first_index = observations[0]
    first_sample = samples[first_index]
    first_observation = first_sample.observation
    source_space = first_sample.source_basis
    target_space = target_basis(armature, entries)
    source_height = first_sample.source_height
    target_height = _target_body_height(armature, entries)
    scale = (
        target_height / source_height
        if source_height > 1.0e-6 and target_height > 1.0e-6
        else 1.0
    )
    scale = min(max(scale, 0.1), 10.0)
    previous_delta = Vector()
    previous_yaw = 0.0
    first_basis = first_sample.source_basis
    canonical_axes = Matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
    )
    first_global_rotation = None
    if first_observation.rotation_matrix is not None:
        values = first_observation.rotation_matrix
        first_global_rotation = (
            canonical_axes
            @ Matrix((values[0:3], values[3:6], values[6:9]))
            @ canonical_axes.inverted_safe()
        )
    source_to_target = target_space @ source_space.inverted_safe()
    target_up = target_space @ Vector((0.0, 0.0, 1.0))
    for index, sample in enumerate(samples):
        observation = sample.observation
        if observation is None or not observation.valid:
            zeros[index] = previous_delta.copy()
            yaws[index] = previous_yaw
            continue
        camera_delta = Vector(observation.translation) - Vector(
            first_observation.translation
        )
        canonical = _canonical_vector(camera_delta)
        previous_delta = (
            target_space @ (source_space.inverted_safe() @ canonical)
        ) * scale
        if (
            observation.rotation_matrix is not None
            and first_global_rotation is not None
        ):
            values = observation.rotation_matrix
            global_rotation = (
                canonical_axes
                @ Matrix((values[0:3], values[3:6], values[6:9]))
                @ canonical_axes.inverted_safe()
            )
            relative_facing = first_global_rotation.inverted_safe() @ global_rotation
            target_relative = (
                source_to_target @ relative_facing @ source_to_target.inverted_safe()
            )
            previous_yaw = _signed_twist_angle(
                target_relative.to_quaternion(), target_up.normalized()
            )
        else:
            relative_facing = first_basis.inverted_safe() @ sample.source_basis
            previous_yaw = relative_facing.to_euler("XYZ").z
        zeros[index] = previous_delta.copy()
        yaws[index] = previous_yaw
    # Keep yaw on a continuous quaternion branch. Temporal Correction owns
    # outlier rejection and smoothing after the final root channel is built.
    for index in range(1, len(yaws)):
        while yaws[index] - yaws[index - 1] > math.pi:
            yaws[index] -= math.tau
        while yaws[index] - yaws[index - 1] < -math.pi:
            yaws[index] += math.tau
    return zeros, yaws, "AUTOMATIC", []


def _set_root_pose(
    target_kind: str,
    target,
    baseline: Matrix,
    delta: Vector,
    yaw: float,
    target_up: Vector,
) -> None:
    location, rotation, scale = baseline.decompose()
    if target_kind == "BONE":
        desired_rotation = Quaternion(target_up.normalized(), yaw) @ rotation
        desired = Matrix.LocRotScale(location + delta, desired_rotation, scale)
        target.matrix = desired
    else:
        world_axis = rotation @ target_up
        desired_rotation = Quaternion(world_axis.normalized(), yaw) @ rotation
        world_delta = baseline.to_3x3() @ delta
        desired = Matrix.LocRotScale(location + world_delta, desired_rotation, scale)
        target.matrix_world = desired
    bpy.context.view_layer.update()


def _grounding_landmark(frame: CanonicalFrame, *names: str) -> Landmark | None:
    points = body_points(frame)
    for name in names:
        point = points.get(name)
        if point is not None and point.valid:
            return point
    return None


def _capture_grounding_frame(
    armature,
    entries,
    frame: CanonicalFrame,
    destination: float,
    root_kind: str,
    root_target,
) -> grounded_cleanup.GroundFrame:
    """Capture evaluated target feet plus provider image/confidence evidence."""
    if root_kind == "BONE":
        root_position = root_target.matrix.translation.copy()

        def convert(value):
            return value.copy()

        def convert_rotation(value):
            return value.copy()
    else:
        root_position = root_target.matrix_world.translation.copy()

        def convert(value):
            return armature.matrix_world @ value

        world_rotation = armature.matrix_world.to_quaternion()

        def convert_rotation(value):
            return (world_rotation @ value).normalized()

    points = {}
    joints = {}
    orientations = {}
    joint_confidence = {}
    pelvis = evaluated_semantic_bone(armature, entries, "PELVIS")
    if pelvis is not None:
        joints["PELVIS"] = convert(pelvis.head)
        orientations["PELVIS"] = convert_rotation(pelvis.matrix.to_quaternion())
    for side, label in (("L", "LEFT"), ("R", "RIGHT")):
        upper = evaluated_semantic_bone(armature, entries, f"UPPER_LEG_{side}")
        lower = evaluated_semantic_bone(armature, entries, f"LOWER_LEG_{side}")
        foot = evaluated_semantic_bone(armature, entries, f"FOOT_{side}")
        toe = evaluated_semantic_bone(armature, entries, f"TOE_{side}")
        if foot is None:
            continue
        heel_position = convert(foot.head)
        toe_position = convert(toe.head if toe is not None else foot.tail)
        foot_direction = toe_position - heel_position
        heel_landmark = _grounding_landmark(frame, f"{label}_HEEL", f"{label}_ANKLE")
        toe_landmark = _grounding_landmark(
            frame, f"{label}_FOOT_INDEX", f"{label}_TOE", f"{label}_ANKLE"
        )
        for role, bone, landmark_names in (
            ("HIP", upper, (f"{label}_HIP",)),
            ("KNEE", lower, (f"{label}_KNEE",)),
            ("ANKLE", foot, (f"{label}_ANKLE",)),
            ("TOE", toe, (f"{label}_FOOT_INDEX", f"{label}_TOE")),
        ):
            if bone is None:
                continue
            key = f"{side}_{role}"
            joints[key] = convert(bone.head)
            orientations[key] = convert_rotation(bone.matrix.to_quaternion())
            landmark = _grounding_landmark(frame, *landmark_names)
            joint_confidence[key] = (
                float(landmark.confidence) if landmark is not None else 0.0
            )
        if heel_landmark is not None:
            points[f"{side}_HEEL"] = grounded_cleanup.ContactPoint(
                position=heel_position,
                image=Vector(heel_landmark.image),
                confidence=float(heel_landmark.confidence),
                direction=foot_direction,
                motion_position=heel_position - root_position,
            )
        if toe_landmark is not None:
            points[f"{side}_TOE"] = grounded_cleanup.ContactPoint(
                position=toe_position,
                image=Vector(toe_landmark.image),
                confidence=float(toe_landmark.confidence),
                direction=foot_direction,
                motion_position=toe_position - root_position,
            )
    observation = frame.root_motion
    return grounded_cleanup.GroundFrame(
        timestamp=float(frame.timestamp_ms) / 1000.0,
        root=root_position,
        points=points,
        joints=joints,
        orientations=orientations,
        joint_confidence=joint_confidence,
        root_confidence=(
            min(max(float(observation.confidence), 0.0), 1.0)
            if observation is not None and observation.valid
            else 0.0
        ),
    )


def _ik_bone_confidences(ik_proposals: Sequence[dict]) -> dict[object, float]:
    """Carry detector confidence onto generated IK controls.

    IK targets and poles are not ordinary TargetPose channels, so they used to
    be captured with an implicit confidence of 1.0. That made Temporal
    Correction trust a noisy wrist target more than the hand observations from
    which it was built.
    """

    result: dict[object, float] = {}
    for proposal in ik_proposals:
        confidence = min(max(float(proposal.get("confidence", 1.0)), 0.0), 1.0)
        for key in ("target", "pole", "rotation_bend", "toe"):
            bone = proposal.get(key)
            if bone is None:
                continue
            result[bone] = min(result.get(bone, 1.0), confidence)
    return result


def _capture_rohm_frame_input(
    armature,
    target_pose: TargetPose,
    root_kind: str,
) -> _RohmFrameInput:
    """Capture the committed frame pose before complete-sequence cleanup.

    Rotations are global rest-relative anatomical transforms.  They are
    deliberately not Blender ``matrix_basis`` values: RoHM converts these
    globals through the official SMPL-X hierarchy before inference.
    """
    from .rohm_correction import SMPLX_SEMANTICS

    result = _RohmFrameInput()
    pelvis = evaluated_semantic_bone(armature, target_pose.entries, "PELVIS")
    if pelvis is None:
        return result
    offset_basis = Matrix.Identity(3)
    if root_kind != "BONE":
        offset_basis = armature.matrix_world.to_3x3()
    anatomy_basis = target_pose.rig_plan.rest_basis.to_3x3()
    posed_root_basis = target_pose.target_basis.to_3x3()
    posed_root_inverse = posed_root_basis.inverted_safe()
    for semantic_id in SMPLX_SEMANTICS:
        bone = evaluated_semantic_bone(armature, target_pose.entries, semantic_id)
        if bone is None:
            continue
        result.joint_offsets[semantic_id] = offset_basis @ (bone.head - pelvis.head)
        rest_rotation = bone.bone.matrix_local.to_3x3()
        global_delta = bone.matrix.to_3x3() @ rest_rotation.inverted_safe()
        anatomical = posed_root_inverse @ global_delta @ anatomy_basis
        rotation = anatomical.to_quaternion().normalized()
        if rotation.w < 0.0:
            rotation.negate()
        result.global_rotations[semantic_id] = rotation
        channel = target_pose.channel_for(semantic_id)
        result.confidence[semantic_id] = (
            min(max(float(channel.confidence), 0.0), 1.0)
            if channel is not None
            else 0.0
        )
    result.body_height = max(_target_body_height(armature, target_pose.entries), 1.0e-4)
    if root_kind != "BONE":
        world_scale = armature.matrix_world.to_scale()
        result.body_height *= max(
            sum(abs(float(value)) for value in world_scale) / 3.0,
            1.0e-4,
        )
    return result


def _baked_frame_from_target(
    armature,
    frame: CanonicalFrame,
    destination: float,
    root_kind: str,
    root_target,
    root_enabled: bool,
    warning_sink: list[str],
    target_pose: TargetPose,
    rohm_input: _RohmFrameInput,
    ik_proposals: list[dict],
    changed_bones: set[object],
    changed_shapes: list[tuple[object, object]],
) -> _BakedFrameSample:
    sample = _BakedFrameSample(
        destination,
        warning_sink,
        timestamp_seconds=float(frame.timestamp_ms) / 1000.0,
        camera_motion_input=camera_motion_correction.input_from_frame(frame),
    )
    channels = {channel.bone: channel for channel in target_pose.bones.values()}
    ik_confidences = _ik_bone_confidences(ik_proposals)
    ik_location = set()
    ik_rotation = set()
    for proposal in ik_proposals:
        if proposal.get("key", (None, None))[0] == "LEG":
            sample.ik_legs[proposal["key"][1]] = proposal
        ik_rotation.add(proposal["target"])
        if not proposal.get("orientation_only"):
            ik_location.add(proposal["target"])
            if proposal.get("rotation_bend") is None:
                ik_location.add(proposal["pole"])
            if proposal.get("rotation_bend") is not None:
                ik_rotation.add(proposal["rotation_bend"])
            if proposal.get("toe") is not None:
                ik_rotation.add(proposal["toe"])
    for bone in changed_bones:
        channel = channels.get(bone)
        location, rotation, scale = bone.matrix_basis.decompose()
        rotation_channel = bool(
            bone in ik_rotation
            or channel is not None
            and (
                channel.desired_rotation is not None
                or channel.direct_basis_rotation is not None
                or channel.hinge_axis_local is not None
            )
        )
        location_channel = bool(
            bone in ik_location or channel is not None and channel.location is not None
        )
        sample.bones[bone.name] = _BakedBoneSample(
            bone=bone,
            location=location.copy(),
            rotation=rotation.normalized(),
            scale=scale.copy(),
            key_location=location_channel,
            key_rotation=rotation_channel,
            semantic_id=channel.semantic_id if channel else "",
            confidence=(
                min(float(channel.confidence), ik_confidences.get(bone, 1.0))
                if channel
                else ik_confidences.get(bone, 1.0)
            ),
            properties=(
                {name: float(bone[name]) for name in channel.properties if name in bone}
                if channel
                else {}
            ),
        )
    if root_enabled:
        if root_kind == "BONE":
            location, rotation, scale = root_target.matrix_basis.decompose()
            observation = frame.root_motion
            sample.bones[root_target.name] = _BakedBoneSample(
                bone=root_target,
                location=location.copy(),
                rotation=rotation.normalized(),
                scale=scale.copy(),
                key_location=True,
                key_rotation=True,
                semantic_id="ROOT",
                confidence=(
                    min(max(float(observation.confidence), 0.0), 1.0)
                    if observation is not None and observation.valid
                    else 0.0
                ),
            )
        else:
            location, rotation, scale = root_target.matrix_basis.decompose()
            sample.root_object = (location.copy(), rotation.normalized(), scale.copy())
    for _obj, block in changed_shapes:
        sample.shapes[(block.as_pointer(), block.name)] = _BakedShapeSample(
            block, float(block.value)
        )
    for warning in target_pose.warnings:
        if warning not in frame.warnings:
            frame.warnings.append(warning)
        if warning not in warning_sink:
            warning_sink.append(warning)
    sample.grounding = _capture_grounding_frame(
        armature,
        target_pose.entries,
        frame,
        destination,
        root_kind,
        root_target,
    )
    sample.body_height = float(rohm_input.body_height)
    sample.rohm_joint_offsets = {
        semantic: value.copy() for semantic, value in rohm_input.joint_offsets.items()
    }
    sample.rohm_rotations = {
        semantic: value.copy()
        for semantic, value in rohm_input.global_rotations.items()
    }
    sample.rohm_confidence = dict(rohm_input.confidence)
    if sample.grounding is not None:
        sample.rohm_root_confidence = float(sample.grounding.root_confidence)
    for name in motion_filter.SOURCE_BODY_JOINTS:
        point = frame.body.get(name)
        if point is None or not point.valid or point.confidence <= 0.0:
            continue
        sample.source_body_3d[name] = Vector(point.position)
        sample.source_body_2d[name] = Vector(point.image)
    return sample


def _iter_capture_baked_frame(
    armature,
    frame: CanonicalFrame,
    destination: float,
    settings,
    session: PostProcessingSession,
    root_kind: str,
    root_target,
    root_enabled: bool,
    postprocess_config,
    rig_plan: rig_planning.RigSolvePlan,
    warning_sink: list[str],
    capture_rohm_input: bool,
) -> Iterator[tuple[str, int, int]]:
    """Solve one frame in independently cancellable main-thread units."""
    frame_module_ids = {
        "SPINE_CORRECTION",
        "SHOULDER_CORRECTION",
        "ARM_AXIS_CORRECTION",
        "LEG_AXIS_CORRECTION",
    }
    module_count = sum(
        bool(module.enabled) and module.module_id in frame_module_ids
        for module in postprocess_config.modules
    )
    total_steps = module_count + 4 + int(capture_rohm_input)
    completed = 1
    yield "Constructing target pose", completed, total_steps
    target_pose = build_target_pose(
        armature,
        frame,
        settings,
        post_process=False,
        session=session,
        rig_plan=rig_plan,
    )
    pipeline = iter_pipeline(
        target_pose,
        postprocess_config,
        session=session,
    )
    while True:
        try:
            display_name, _index, _total = next(pipeline)
        except StopIteration:
            break
        completed += 1
        yield f"Postprocess: {display_name}", completed, total_steps

    completed += 1
    yield "Preparing IK controls", completed, total_steps
    ik_proposals = _prepare_ik_output(target_pose)
    completed += 1
    yield "Committing pose", completed, total_steps
    commit = _iter_commit_target_pose(target_pose, ik_proposals)
    while True:
        try:
            commit_stage, _commit_current, _commit_total = next(commit)
        except StopIteration as stopped:
            changed_bones, changed_shapes = stopped.value
            break
        yield f"Committing pose: {commit_stage}", completed, total_steps
    rohm_input = _RohmFrameInput()
    if capture_rohm_input:
        completed += 1
        yield "Capturing AI cleanup input", completed, total_steps
        rohm_input = _capture_rohm_frame_input(armature, target_pose, root_kind)
    completed += 1
    yield "Capturing solved channels", completed, total_steps
    return _baked_frame_from_target(
        armature,
        frame,
        destination,
        root_kind,
        root_target,
        root_enabled,
        warning_sink,
        target_pose,
        rohm_input,
        ik_proposals,
        changed_bones,
        changed_shapes,
    )


def _sequence_timestamps(samples: Sequence[_BakedFrameSample]) -> list[float]:
    return [float(sample.timestamp_seconds) for sample in samples]


def _continuous_runs(
    indices: Sequence[int],
    timestamps: Sequence[float],
) -> list[list[int]]:
    intervals = [
        timestamps[right] - timestamps[left]
        for left, right in zip(
            range(len(timestamps)), range(1, len(timestamps)), strict=False
        )
        if timestamps[right] > timestamps[left]
    ]
    normal = float(sorted(intervals)[len(intervals) // 2]) if intervals else 0.0
    runs: list[list[int]] = []
    for index in indices:
        discontinuity = not runs or index != runs[-1][-1] + 1
        if not discontinuity:
            delta = timestamps[index] - timestamps[runs[-1][-1]]
            discontinuity = delta <= 0.0 or normal > 0.0 and delta > 2.5 * normal
        if discontinuity:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def _fill_short_bone_gaps(
    samples: list[_BakedFrameSample],
    timestamps: Sequence[float],
) -> None:
    names = {name for sample in samples for name in sample.bones}
    for name in names:
        present = [
            index for index, sample in enumerate(samples) if name in sample.bones
        ]
        for left, right in zip(present, present[1:], strict=False):
            gap = right - left - 1
            if gap < 1:
                continue
            span = timestamps[right] - timestamps[left]
            local_steps = [
                timestamps[index + 1] - timestamps[index]
                for index in range(max(0, left - 2), min(len(samples) - 1, right + 2))
                if timestamps[index + 1] > timestamps[index]
            ]
            normal = sorted(local_steps)[len(local_steps) // 2] if local_steps else span
            if span <= 0.0 or span > 0.5 or span > 2.5 * normal * (gap + 1):
                continue
            first, last = samples[left].bones[name], samples[right].bones[name]
            if not (first.key_rotation or first.key_location):
                continue
            # A short hole is safe to bridge only when it is bounded by two
            # credible observations. Otherwise interpolation would turn a
            # weak detection into apparently authoritative animation.
            if min(first.confidence, last.confidence) < 0.35:
                continue
            for offset in range(1, gap + 1):
                factor = min(
                    max(
                        (timestamps[left + offset] - timestamps[left]) / span,
                        0.0,
                    ),
                    1.0,
                )
                samples[left + offset].bones[name] = _BakedBoneSample(
                    bone=first.bone,
                    location=first.location.lerp(last.location, factor),
                    rotation=first.rotation.slerp(last.rotation, factor).normalized(),
                    scale=first.scale.copy(),
                    key_location=first.key_location and last.key_location,
                    key_rotation=first.key_rotation and last.key_rotation,
                    semantic_id=first.semantic_id or last.semantic_id,
                    confidence=min(first.confidence, last.confidence),
                    properties={
                        name: float(first.properties[name])
                        + (float(last.properties[name]) - float(first.properties[name]))
                        * factor
                        for name in first.properties.keys() & last.properties.keys()
                    },
                )


def _continuous_property(name: str) -> bool:
    lowered = name.casefold()
    return not any(
        token in lowered
        for token in ("switch", "ik_fk", "fk_ik", "visibility", "mute", "owner")
    )


def _project_temporal_finger_hinges(
    armature,
    samples: list[_BakedFrameSample],
) -> int:
    """Reapply anatomical finger limits after independent channel filtering."""

    corrected = 0
    temporary = _sequence_state(armature)
    try:
        for sample in samples:
            _scene_set_float(bpy.context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample)
            by_semantic = {
                channel.semantic_id: channel
                for channel in sample.bones.values()
                if channel.semantic_id
            }
            for side in ("L", "R"):
                hand_channel = by_semantic.get(f"HAND_{side}")
                if hand_channel is None:
                    continue
                hand = hand_channel.bone
                first_channels = [
                    by_semantic.get(f"{finger}_01_{side}")
                    for finger in ("INDEX", "MIDDLE", "RING", "PINKY")
                ]
                if any(channel is None for channel in first_channels):
                    continue
                rest_hand = hand.bone.matrix_local.to_3x3()
                hand_delta = (
                    (hand.matrix.to_3x3() @ rest_hand.inverted_safe())
                    .to_quaternion()
                    .normalized()
                )
                primary = (
                    sum(
                        (channel.bone.bone.head_local for channel in first_channels),
                        Vector(),
                    )
                    / len(first_channels)
                    - hand.bone.head_local
                )
                lateral = (
                    first_channels[0].bone.bone.head_local
                    - first_channels[-1].bone.bone.head_local
                )
                normal = primary.cross(lateral)
                if normal.length <= 1.0e-8:
                    continue
                normal.normalize()
                for finger in ("INDEX", "MIDDLE", "RING", "PINKY"):
                    parent_delta = hand_delta
                    for segment in range(1, 4):
                        channel = by_semantic.get(f"{finger}_{segment:02d}_{side}")
                        if channel is None:
                            break
                        bone = channel.bone
                        rest_direction = (
                            bone.bone.tail_local - bone.bone.head_local
                        ).normalized()
                        current_direction = (bone.tail - bone.head).normalized()
                        limit = _FINGER_ARTICULATION_LIMITS["FINGER"][segment - 1]
                        desired_delta, constrained = (
                            _bounded_parent_relative_direction_delta(
                                rest_direction,
                                current_direction,
                                parent_delta,
                                limit,
                                bend_reference=normal,
                                maximum_off_plane=math.radians(
                                    35.0 if segment == 1 else 25.0
                                ),
                                flexion_bounds=(
                                    (
                                        math.radians(-25.0 if segment == 1 else -15.0),
                                        limit,
                                    )
                                    if side == "L"
                                    else (
                                        -limit,
                                        math.radians(25.0 if segment == 1 else 15.0),
                                    )
                                ),
                            )
                        )
                        if constrained:
                            rest_rotation = bone.bone.matrix_local.to_3x3()
                            desired_rotation = (
                                (desired_delta.to_matrix() @ rest_rotation)
                                .to_quaternion()
                                .normalized()
                            )
                            bone.matrix = Matrix.LocRotScale(
                                bone.matrix.translation,
                                desired_rotation,
                                bone.matrix.to_scale(),
                            )
                            bpy.context.view_layer.update()
                            _store_bone_basis(sample, bone)
                            corrected += 1
                        parent_delta = desired_delta
    finally:
        _restore_sequence_state(armature, temporary)
    return corrected


def _iter_temporal_channels(
    armature,
    samples: list[_BakedFrameSample],
    settings,
) -> Iterator[SequenceApplyProgress]:
    timestamps = _sequence_timestamps(samples)
    smooth_motion = bool(settings.smooth_motion)
    strength = float(settings.smoothing_strength)
    stabilize_hands = bool(settings.stabilize_hands)
    finger_maximum_speed = float(settings.finger_max_angular_speed)
    wrist_maximum_speed = float(settings.wrist_max_angular_speed)
    hand_reacquire_delay = float(settings.hand_reacquire_delay)
    _fill_short_bone_gaps(samples, timestamps)
    names = {name for sample in samples for name in sample.bones}
    shape_keys = {key for sample in samples for key in sample.shapes}
    properties = {
        (name, property_name)
        for sample in samples
        for name, channel in sample.bones.items()
        for property_name in channel.properties
        if _continuous_property(property_name)
    }
    object_indices = [
        index for index, sample in enumerate(samples) if sample.root_object
    ]
    total = len(names) + len(shape_keys) + len(properties) + bool(object_indices)
    completed = 0
    repaired = 0
    low_confidence = 0
    for name in sorted(names):
        location_indices = [
            index
            for index, sample in enumerate(samples)
            if name in sample.bones and sample.bones[name].key_location
        ]
        rotation_indices = [
            index
            for index, sample in enumerate(samples)
            if name in sample.bones and sample.bones[name].key_rotation
        ]
        for run in _continuous_runs(location_indices, timestamps):
            channels = [samples[index].bones[name] for index in run]
            run_times = [timestamps[index] for index in run]
            confidences = [float(channel.confidence) for channel in channels]
            low_confidence += sum(value < 0.2 for value in confidences)
            values = [channel.location.copy() for channel in channels]
            if smooth_motion:
                values = animation_cleanup.smooth_vectors(
                    values, run_times, confidences, strength
                )
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            for channel, value in zip(channels, values, strict=True):
                channel.location = value
        for run in _continuous_runs(rotation_indices, timestamps):
            channels = [samples[index].bones[name] for index in run]
            run_times = [timestamps[index] for index in run]
            confidences = [float(channel.confidence) for channel in channels]
            low_confidence += sum(value < 0.2 for value in confidences)
            values = [channel.rotation.copy() for channel in channels]
            semantic_id = next(
                (channel.semantic_id for channel in channels if channel.semantic_id),
                "",
            )
            is_finger = semantic_id.startswith(
                ("THUMB_", "INDEX_", "MIDDLE_", "RING_", "PINKY_")
            )
            is_wrist = semantic_id.startswith("HAND_")
            if stabilize_hands and (is_finger or is_wrist):
                values, hold_count = animation_cleanup.stabilize_quaternion_jumps(
                    values,
                    run_times,
                    finger_maximum_speed if is_finger else wrist_maximum_speed,
                    reacquire_seconds=hand_reacquire_delay,
                )
                repaired += hold_count
                yield SequenceApplyProgress(
                    "Stabilizing hand rotations", completed, total
                )
            if smooth_motion and not (stabilize_hands and is_finger):
                smoother = (
                    animation_cleanup.smooth_quaternions_bounded
                    if is_wrist
                    else animation_cleanup.smooth_quaternions
                )
                values = smoother(values, run_times, confidences, strength)
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            if stabilize_hands and (is_finger or is_wrist):
                values, speed_count = animation_cleanup.limit_quaternion_speed(
                    values,
                    run_times,
                    finger_maximum_speed if is_finger else wrist_maximum_speed,
                )
                repaired += speed_count
            for channel, value in zip(channels, values, strict=True):
                channel.rotation = value
        completed += 1

    for key in sorted(shape_keys):
        indices = [
            index for index, sample in enumerate(samples) if key in sample.shapes
        ]
        for left, right in zip(indices, indices[1:], strict=False):
            gap = right - left - 1
            if gap < 1 or gap > 2:
                continue
            if len(_continuous_runs(list(range(left, right + 1)), timestamps)) != 1:
                continue
            first, last = samples[left].shapes[key], samples[right].shapes[key]
            for offset in range(1, gap + 1):
                span = timestamps[right] - timestamps[left]
                factor = (
                    (timestamps[left + offset] - timestamps[left]) / span
                    if span > 1.0e-8
                    else offset / (gap + 1)
                )
                samples[left + offset].shapes[key] = _BakedShapeSample(
                    first.block,
                    first.value + (last.value - first.value) * factor,
                )
        indices = [
            index for index, sample in enumerate(samples) if key in sample.shapes
        ]
        for run in _continuous_runs(indices, timestamps):
            values = [samples[index].shapes[key].value for index in run]
            run_times = [timestamps[index] for index in run]
            if smooth_motion:
                values = animation_cleanup.smooth_scalars(
                    values, run_times, None, strength
                )
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            block = samples[run[0]].shapes[key].block
            minimum = float(getattr(block, "slider_min", 0.0))
            maximum = float(getattr(block, "slider_max", 1.0))
            for index, value in zip(run, values, strict=True):
                samples[index].shapes[key].value = min(max(value, minimum), maximum)
        completed += 1

    for name, property_name in sorted(properties):
        indices = [
            index
            for index, sample in enumerate(samples)
            if name in sample.bones and property_name in sample.bones[name].properties
        ]
        for run in _continuous_runs(indices, timestamps):
            channels = [samples[index].bones[name] for index in run]
            values = [float(channel.properties[property_name]) for channel in channels]
            run_times = [timestamps[index] for index in run]
            confidences = [float(channel.confidence) for channel in channels]
            if smooth_motion:
                values = animation_cleanup.smooth_scalars(
                    values, run_times, confidences, strength
                )
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            for channel, value in zip(channels, values, strict=True):
                try:
                    limits = channel.bone.id_properties_ui(property_name).as_dict()
                except (AttributeError, KeyError, TypeError):
                    limits = {}
                if "min" in limits:
                    value = max(value, float(limits["min"]))
                if "max" in limits:
                    value = min(value, float(limits["max"]))
                channel.properties[property_name] = value
        completed += 1

    for run in _continuous_runs(object_indices, timestamps):
        run_times = [timestamps[index] for index in run]
        locations = [samples[index].root_object[0].copy() for index in run]
        rotations = [samples[index].root_object[1].copy() for index in run]
        confidences = [
            float(samples[index].grounding.root_confidence)
            if samples[index].grounding is not None
            else 1.0
            for index in run
        ]
        if smooth_motion:
            locations = animation_cleanup.smooth_vectors(
                locations, run_times, confidences, strength
            )
            rotations = animation_cleanup.smooth_quaternions(
                rotations, run_times, confidences, strength
            )
            yield SequenceApplyProgress("Smoothing animation", completed, total)
        for index, location, rotation in zip(run, locations, rotations, strict=True):
            _old_location, _old_rotation, scale = samples[index].root_object
            samples[index].root_object = (location, rotation, scale)
    if object_indices:
        completed += 1

    if stabilize_hands:
        anatomical_count = _project_temporal_finger_hinges(armature, samples)
        repaired += anatomical_count
        if anatomical_count:
            yield SequenceApplyProgress(
                "Projecting anatomical finger hinges", total, total
            )

    if repaired:
        _append_bake_warning(
            samples,
            f"Temporal Correction repaired {repaired} noisy motion sample(s)",
        )
    if low_confidence > max(6, len(samples)):
        _append_bake_warning(
            samples,
            "Temporal Correction reduced the influence of low-confidence motion samples",
        )


def _apply_baked_sample_pose(armature, sample: _BakedFrameSample) -> None:
    if sample.root_object is not None:
        location, rotation, scale = sample.root_object
        armature.matrix_basis = Matrix.LocRotScale(location, rotation, scale)
    for channel in sorted(
        sample.bones.values(), key=lambda item: len(item.bone.parent_recursive)
    ):
        channel.bone.matrix_basis = Matrix.LocRotScale(
            channel.location, channel.rotation, channel.scale
        )
        for name, value in channel.properties.items():
            channel.bone[name] = value
    armature.update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()


def _store_bone_basis(sample: _BakedFrameSample, bone) -> None:
    channel = sample.bones.get(bone.name)
    if channel is None:
        return
    location, rotation, scale = bone.matrix_basis.decompose()
    channel.location = location.copy()
    channel.rotation = rotation.normalized()
    channel.scale = scale.copy()


def _contact_targets(
    result: grounded_cleanup.GroundingResult,
    timestamps: Sequence[float],
) -> list[dict[str, tuple[Vector, float]]]:
    targets = [dict() for _ in timestamps]
    for segment in result.segments:
        duration = max(timestamps[segment.end] - timestamps[segment.start], 1.0e-4)
        transition = max(2, int(round(0.1 * len(segment.weights) / duration)))
        for offset, index in enumerate(range(segment.start, segment.end + 1)):
            edge = min(offset + 1, segment.end - index + 1) / max(transition, 1)
            weight = edge * edge * (3.0 - 2.0 * min(max(edge, 0.0), 1.0))
            weight *= max(result.contacts[segment.point][index], 0.35)
            targets[index][segment.point] = (segment.anchor.copy(), min(weight, 1.0))
    return targets


def _reduce_ground_contact_outliers(
    frames: Sequence[grounded_cleanup.GroundFrame],
) -> int:
    """Clean isolated evaluated heel/toe failures before contact analysis."""
    timestamps = [float(frame.timestamp) for frame in frames]
    repaired = 0
    names = {name for frame in frames for name in frame.points}
    for name in names:
        indices = [index for index, frame in enumerate(frames) if name in frame.points]
        for run in _continuous_runs(indices, timestamps):
            points = [frames[index].points[name] for index in run]
            values, count = animation_cleanup.reduce_vector_outliers(
                [point.position for point in points],
                [timestamps[index] for index in run],
                [point.confidence for point in points],
            )
            repaired += count
            for index, point, value in zip(run, points, values, strict=True):
                point.position = value
                point.motion_position = value - frames[index].root
    return repaired


def _analytic_two_bone_joint(
    hip: Vector,
    current_knee: Vector,
    desired: Vector,
    upper_length: float,
    lower_length: float,
    target_up: Vector,
) -> tuple[Vector | None, Vector, bool]:
    axis = desired - hip
    if axis.length < 1.0e-7 or upper_length <= 0.0 or lower_length <= 0.0:
        return None, desired.copy(), False
    unreachable = axis.length >= (upper_length + lower_length) * 0.998
    distance = min(
        max(axis.length, abs(upper_length - lower_length) + 1.0e-5),
        (upper_length + lower_length) * 0.995,
    )
    axis.normalize()
    bend = current_knee - hip
    bend -= axis * bend.dot(axis)
    if bend.length < 1.0e-6:
        bend = target_up.cross(axis)
    if bend.length < 1.0e-6:
        bend = Vector((1.0, 0.0, 0.0)).cross(axis)
    bend.normalize()
    along = (
        upper_length * upper_length + distance * distance - lower_length * lower_length
    ) / (2.0 * distance)
    perpendicular = math.sqrt(max(upper_length * upper_length - along * along, 0.0))
    desired_knee = hip + axis * along + bend * perpendicular
    clamped_end = hip + axis * distance
    return desired_knee, clamped_end, unreachable


def _repair_fk_leg(
    sample: _BakedFrameSample,
    upper,
    lower,
    foot,
    desired_ankle: Vector,
    weight: float,
    target_up: Vector,
) -> bool:
    hip = upper.head.copy()
    current_knee = lower.head.copy()
    current_ankle = foot.head.copy()
    desired = current_ankle.lerp(desired_ankle, min(max(weight, 0.0), 1.0))
    desired_knee, desired, unreachable = _analytic_two_bone_joint(
        hip,
        current_knee,
        desired,
        float(upper.bone.length),
        float(lower.bone.length),
        target_up,
    )
    if desired_knee is None:
        return False
    current_upper = upper.tail - upper.head
    desired_upper = desired_knee - hip
    if current_upper.length > 1.0e-8 and desired_upper.length > 1.0e-8:
        rotation = current_upper.rotation_difference(desired_upper)
        matrix = upper.matrix.copy()
        matrix = Matrix.LocRotScale(
            matrix.translation,
            (rotation @ matrix.to_quaternion()).normalized(),
            matrix.to_scale(),
        )
        upper.matrix = matrix
        bpy.context.view_layer.update()
    current_lower = lower.tail - lower.head
    desired_lower = desired - lower.head
    if current_lower.length > 1.0e-8 and desired_lower.length > 1.0e-8:
        rotation = current_lower.rotation_difference(desired_lower)
        matrix = lower.matrix.copy()
        matrix = Matrix.LocRotScale(
            matrix.translation,
            (rotation @ matrix.to_quaternion()).normalized(),
            matrix.to_scale(),
        )
        lower.matrix = matrix
        bpy.context.view_layer.update()
    _store_bone_basis(sample, upper)
    _store_bone_basis(sample, lower)
    return unreachable


def _append_bake_warning(
    samples: Sequence[_BakedFrameSample],
    warning: str,
) -> None:
    if samples and warning not in samples[0].warning_sink:
        samples[0].warning_sink.append(warning)


def _iter_ground_contacts(
    armature,
    entries,
    samples: list[_BakedFrameSample],
    strength: float,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_up: Vector,
) -> Iterator[SequenceApplyProgress]:
    timestamps = _sequence_timestamps(samples)
    for run in _continuous_runs(list(range(len(samples))), timestamps):
        yield from _iter_ground_contacts_run(
            armature,
            entries,
            [samples[index] for index in run],
            strength,
            effective_mode,
            root_kind,
            root_target,
            target_up,
        )


def _iter_ground_contacts_run(
    armature,
    entries,
    samples: list[_BakedFrameSample],
    strength: float,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_up: Vector,
) -> Iterator[SequenceApplyProgress]:
    ground_frames = [sample.grounding for sample in samples]
    if not ground_frames or any(frame is None for frame in ground_frames):
        warning = (
            "Ground Contacts skipped because evaluated foot samples were unavailable"
        )
        _append_bake_warning(samples, warning)
        return
    body_height = max(_target_body_height(armature, entries), 1.0e-4)
    grounding_up = (
        target_up.normalized()
        if root_kind == "BONE"
        else (armature.matrix_world.to_3x3() @ target_up).normalized()
    )
    grounding_height = body_height
    if root_kind != "BONE":
        world_scale = armature.matrix_world.to_scale()
        grounding_height *= max(sum(abs(value) for value in world_scale) / 3.0, 1.0e-4)
    contact_repairs = _reduce_ground_contact_outliers(ground_frames)
    if contact_repairs:
        _append_bake_warning(
            samples,
            f"Ground Contacts repaired {contact_repairs} isolated foot sample(s)",
        )
    result = grounded_cleanup.analyze_grounding(
        ground_frames, grounding_height, grounding_up
    )
    available_points = {name for frame in ground_frames for name in frame.points}
    missing_sides = [
        side
        for side in ("L", "R")
        if f"{side}_HEEL" not in available_points
        and f"{side}_TOE" not in available_points
    ]
    if missing_sides:
        warning = "Ground Contacts limited by missing foot mapping(s): " + ", ".join(
            missing_sides
        )
        _append_bake_warning(samples, warning)
    for warning in result.warnings:
        _append_bake_warning(samples, warning)
    for index in range(len(samples)):
        yield SequenceApplyProgress("Detecting foot contacts", index + 1, len(samples))

    if effective_mode == "AUTOMATIC":
        latest_roots = [frame.root.copy() for frame in ground_frames]
        iterator = grounded_cleanup.stabilize_root(
            ground_frames,
            result.contacts,
            result.segments,
            grounding_height,
            grounding_up,
            strength,
        )
        for iteration, roots in iterator:
            latest_roots = roots
            yield SequenceApplyProgress("Stabilizing root motion", iteration, 10)
        temporary = _sequence_state(armature)
        try:
            for index, (sample, desired_root) in enumerate(
                zip(samples, latest_roots, strict=False), 1
            ):
                _scene_set_float(bpy.context.scene, sample.destination)
                _apply_baked_sample_pose(armature, sample)
                if root_kind == "BONE":
                    matrix = root_target.matrix.copy()
                    matrix.translation = desired_root
                    root_target.matrix = matrix
                    bpy.context.view_layer.update()
                    _store_bone_basis(sample, root_target)
                else:
                    matrix = root_target.matrix_world.copy()
                    matrix.translation = desired_root
                    root_target.matrix_world = matrix
                    bpy.context.view_layer.update()
                    location, rotation, scale = root_target.matrix_basis.decompose()
                    sample.root_object = (
                        location.copy(),
                        rotation.normalized(),
                        scale.copy(),
                    )
                yield SequenceApplyProgress(
                    "Applying stabilized root", index, len(samples)
                )
        finally:
            _restore_sequence_state(armature, temporary)

    targets = _contact_targets(result, [frame.timestamp for frame in ground_frames])
    ownership = limb_ownership(armature)
    temporary = _sequence_state(armature)
    unreachable_count = 0
    try:
        for index, (sample, frame_targets) in enumerate(
            zip(samples, targets, strict=False), 1
        ):
            _scene_set_float(bpy.context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample)
            translation_solver = PoseTranslationSolver()
            for side in ("L", "R"):
                heel_target = frame_targets.get(f"{side}_HEEL")
                toe_target = frame_targets.get(f"{side}_TOE")
                if heel_target is None and toe_target is None:
                    continue
                upper = evaluated_semantic_bone(armature, entries, f"UPPER_LEG_{side}")
                lower = evaluated_semantic_bone(armature, entries, f"LOWER_LEG_{side}")
                foot = evaluated_semantic_bone(armature, entries, f"FOOT_{side}")
                toe = evaluated_semantic_bone(armature, entries, f"TOE_{side}")
                if upper is None or lower is None or foot is None:
                    continue
                to_pose = (
                    (lambda value: value.copy())
                    if root_kind == "BONE"
                    else (lambda value: armature.matrix_world.inverted_safe() @ value)
                )
                current_heel = foot.head.copy()
                current_toe = toe.head.copy() if toe is not None else foot.tail.copy()
                preserved_foot_rotation = foot.matrix.to_quaternion().normalized()
                if heel_target is not None:
                    desired_ankle = to_pose(heel_target[0])
                    weight = heel_target[1]
                else:
                    desired_ankle = to_pose(toe_target[0]) - (
                        current_toe - current_heel
                    )
                    weight = toe_target[1]
                if (desired_ankle - upper.head).length >= (
                    float(upper.bone.length) + float(lower.bone.length)
                ) * 0.998:
                    unreachable_count += 1
                info = ownership.get(("LEG", side), {})
                if info.get("mode") == "IK" and info.get("target") is not None:
                    target = info["target"]
                    desired_joint, clamped_ankle, _unreachable = (
                        _analytic_two_bone_joint(
                            upper.head.copy(),
                            lower.head.copy(),
                            current_heel.lerp(desired_ankle, weight),
                            float(upper.bone.length),
                            float(lower.bone.length),
                            target_up,
                        )
                    )
                    desired_endpoint = (
                        clamped_ankle if desired_joint is not None else desired_ankle
                    )
                    for _attempt in range(3):
                        error = desired_endpoint - foot.head
                        if error.length < 1.0e-5:
                            break
                        matrix = target.matrix.copy()
                        matrix.translation += error
                        target.matrix = matrix
                        bpy.context.view_layer.update()
                    pole = info.get("pole")
                    rotation_bend = info.get("rotation_bend")
                    proposal = sample.ik_legs.get(side, {})
                    if (
                        pole is not None
                        and desired_joint is not None
                        and rotation_bend is None
                    ):
                        pole_location = _ik_pole_location(
                            Matrix.Translation(upper.head),
                            Matrix.Translation(desired_joint),
                            Matrix.Translation(desired_endpoint),
                            pole,
                            proposal.get("pole_rest_geometry"),
                        )
                        _set_pose_translation_preserve_basis_rotation(
                            pole,
                            pole_location,
                            solver=translation_solver,
                        )
                        _store_bone_basis(sample, pole)
                    elif rotation_bend is not None and desired_joint is not None:
                        axis = desired_endpoint - upper.head
                        if axis.length > 1.0e-8:
                            axis.normalize()
                            current_bend = upper.tail - (
                                upper.head + axis * (upper.tail - upper.head).dot(axis)
                            )
                            wanted_bend = desired_joint - (
                                upper.head
                                + axis * (desired_joint - upper.head).dot(axis)
                            )
                            if (
                                current_bend.length > 1.0e-8
                                and wanted_bend.length > 1.0e-8
                            ):
                                current_bend.normalize()
                                wanted_bend.normalize()
                                angle = math.atan2(
                                    axis.dot(current_bend.cross(wanted_bend)),
                                    current_bend.dot(wanted_bend),
                                )
                                rotation = rotation_bend.rotation_euler.copy()
                                rotation.y += angle
                                rotation_bend.rotation_euler = rotation
                                bpy.context.view_layer.update()
                                _store_bone_basis(sample, rotation_bend)
                    for _attempt in range(2):
                        error = desired_endpoint - foot.head
                        if error.length < 1.0e-5:
                            break
                        matrix = target.matrix.copy()
                        matrix.translation += error
                        target.matrix = matrix
                        bpy.context.view_layer.update()
                    _store_bone_basis(sample, target)
                else:
                    _repair_fk_leg(
                        sample, upper, lower, foot, desired_ankle, weight, target_up
                    )
                    # Ground contact owns the planted position, not the foot's
                    # orientation. Counter-rotate the FK foot after moving its
                    # parent leg so the captured foot direction remains intact.
                    # IK effectors above are translated without changing their
                    # rotation. This avoids heel/toe contact evidence selecting
                    # a sudden, visually unrelated orientation branch.
                    matrix = foot.matrix.copy()
                    foot.matrix = Matrix.LocRotScale(
                        matrix.translation,
                        preserved_foot_rotation,
                        matrix.to_scale(),
                    )
                    bpy.context.view_layer.update()
                    _store_bone_basis(sample, foot)
            yield SequenceApplyProgress("Repairing planted feet", index, len(samples))
    finally:
        _restore_sequence_state(armature, temporary)
    if unreachable_count:
        warning = f"Clamped {unreachable_count} unreachable planted-leg correction(s) without stretching"
        _append_bake_warning(samples, warning)


def _drain(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


def _has_groundable_body_channels(samples: Sequence[_BakedFrameSample]) -> bool:
    return any(
        channel.semantic_id.startswith(
            ("PELVIS", "UPPER_LEG_", "LOWER_LEG_", "FOOT_", "TOE_")
        )
        for sample in samples
        for channel in sample.bones.values()
    )


def _publish_camera_motion_diagnostics(item, result, track_status: str) -> None:
    if item is None:
        return
    try:
        item.camera_track_status = str(track_status)
        item.camera_tracked_frames = int(result.tracked_frames)
        item.camera_rejected_frames = int(result.rejected_frames)
        item.camera_median_inliers = float(result.median_inliers)
        item.camera_corrected_intervals = int(result.corrected_intervals)
        item.last_warning = str(result.warning)
        item.status = "WARNING" if result.warning else "VALID"
    except (AttributeError, ReferenceError, RuntimeError):
        pass


def _translate_grounding(frame, delta: Vector) -> None:
    if frame is None or delta.length_squared <= 1.0e-20:
        return
    frame.root += delta
    for point in frame.points.values():
        point.position += delta
    for name in tuple(frame.joints):
        frame.joints[name] += delta


def _iter_camera_motion_correction(
    armature,
    samples: list[_BakedFrameSample],
    settings,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_lateral: Vector,
    object_delta_basis: Matrix,
    source_to_target_scale: float,
    track: camera_motion_correction.CameraMotionTrack | None,
    ui_settings,
    track_status: str,
    is_video: bool,
) -> Iterator[SequenceApplyProgress]:
    inputs = [sample.camera_motion_input for sample in samples]
    usable_inputs = [item for item in inputs if item is not None]
    if (
        not is_video
        or effective_mode != "AUTOMATIC"
        or track is None
        or len(usable_inputs) != len(samples)
    ):
        warning = ""
        if is_video and effective_mode == "AUTOMATIC" and track is None:
            warning = (
                "Camera Motion Correction skipped: no matching camera track and "
                "the source video was unavailable"
            )
            _append_bake_warning(samples, warning)
        result = camera_motion_correction.CameraCorrectionResult(
            offsets_x=tuple(0.0 for _sample in samples),
            corrected_intervals=0,
            corrected_frames=0,
            tracked_frames=0,
            rejected_frames=len(samples) if effective_mode == "AUTOMATIC" else 0,
            median_inliers=0.0,
            warning=warning,
        )
        _publish_camera_motion_diagnostics(ui_settings, result, track_status)
        return

    result = camera_motion_correction.correction_offsets(
        usable_inputs,
        track,
        strength=float(settings.lateral_root_strength),
    )
    _publish_camera_motion_diagnostics(ui_settings, result, track_status)
    if result.warning:
        _append_bake_warning(samples, result.warning)
    if not result.corrected_frames:
        return

    temporary = _sequence_state(armature)
    try:
        for index, (sample, offset) in enumerate(
            zip(samples, result.offsets_x, strict=True), 1
        ):
            if abs(float(offset)) <= 1.0e-12:
                yield SequenceApplyProgress(
                    "Camera Motion: applying lateral root correction",
                    index,
                    len(samples),
                )
                continue
            _scene_set_float(bpy.context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample)
            target_delta = target_lateral * (
                float(offset) * float(source_to_target_scale)
            )
            applied_delta = target_delta
            if root_kind == "BONE":
                original = sample.bones.get(root_target.name)
                if original is None:
                    continue
                original_rotation = original.rotation.copy()
                original_scale = original.scale.copy()
                matrix = root_target.matrix.copy()
                matrix.translation += target_delta
                root_target.matrix = matrix
                bpy.context.view_layer.update()
                _store_bone_basis(sample, root_target)
                stored = sample.bones[root_target.name]
                stored.rotation = original_rotation
                stored.scale = original_scale
            else:
                if sample.root_object is None:
                    continue
                applied_delta = object_delta_basis @ target_delta
                original_rotation = sample.root_object[1].copy()
                original_scale = sample.root_object[2].copy()
                matrix = root_target.matrix_world.copy()
                matrix.translation += applied_delta
                root_target.matrix_world = matrix
                bpy.context.view_layer.update()
                location, _rotation, _scale = root_target.matrix_basis.decompose()
                sample.root_object = (
                    location.copy(),
                    original_rotation,
                    original_scale,
                )
            _translate_grounding(sample.grounding, applied_delta)
            yield SequenceApplyProgress(
                "Camera Motion: applying lateral root correction",
                index,
                len(samples),
            )
    finally:
        _restore_sequence_state(armature, temporary)
    _append_bake_warning(
        samples,
        "Camera Motion Correction adjusted lateral root travel in "
        f"{result.corrected_intervals} interval(s) without changing rotations",
    )


def _iter_apply_rohm_root_deltas(
    armature,
    samples: Sequence[_BakedFrameSample],
    root_deltas: Sequence[Sequence[float]],
    root_kind: str,
    root_target,
) -> Iterator[SequenceApplyProgress]:
    if len(root_deltas) != len(samples):
        raise BackendError("RoHM root corrections do not match the baked sequence")
    temporary = _sequence_state(armature)
    try:
        for index, (sample, raw_delta) in enumerate(
            zip(samples, root_deltas, strict=True), 1
        ):
            delta = Vector(tuple(float(value) for value in raw_delta))
            if delta.length_squared <= 1.0e-20:
                yield SequenceApplyProgress(
                    "RoHM: applying corrected root", index, len(samples)
                )
                continue
            _scene_set_float(bpy.context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample)
            if root_kind == "BONE":
                original = sample.bones.get(root_target.name)
                if original is None:
                    yield SequenceApplyProgress(
                        "RoHM: applying corrected root", index, len(samples)
                    )
                    continue
                original_rotation = original.rotation.copy()
                original_scale = original.scale.copy()
                matrix = root_target.matrix.copy()
                matrix.translation += delta
                root_target.matrix = matrix
                bpy.context.view_layer.update()
                _store_bone_basis(sample, root_target)
                stored = sample.bones[root_target.name]
                stored.rotation = original_rotation
                stored.scale = original_scale
            else:
                if sample.root_object is None:
                    yield SequenceApplyProgress(
                        "RoHM: applying corrected root", index, len(samples)
                    )
                    continue
                original_rotation = sample.root_object[1].copy()
                original_scale = sample.root_object[2].copy()
                matrix = root_target.matrix_world.copy()
                matrix.translation += delta
                root_target.matrix_world = matrix
                bpy.context.view_layer.update()
                location, _rotation, _scale = root_target.matrix_basis.decompose()
                sample.root_object = (
                    location.copy(),
                    original_rotation,
                    original_scale,
                )
            _translate_grounding(sample.grounding, delta)
            yield SequenceApplyProgress(
                "RoHM: applying corrected root", index, len(samples)
            )
    finally:
        _restore_sequence_state(armature, temporary)


def _merge_rohm_pose_into_sample(
    sample: _BakedFrameSample,
    target_pose: TargetPose,
    ik_proposals: Sequence[dict],
    changed_bones: set[object],
) -> None:
    channels = {channel.bone: channel for channel in target_pose.bones.values()}
    ik_confidences = _ik_bone_confidences(ik_proposals)
    ik_location = set()
    ik_rotation = set()
    for proposal in ik_proposals:
        ik_rotation.add(proposal["target"])
        if not proposal.get("orientation_only"):
            ik_location.add(proposal["target"])
            if proposal.get("rotation_bend") is None:
                ik_location.add(proposal["pole"])
            if proposal.get("rotation_bend") is not None:
                ik_rotation.add(proposal["rotation_bend"])
            if proposal.get("toe") is not None:
                ik_rotation.add(proposal["toe"])
    for bone in changed_bones:
        channel = channels.get(bone)
        previous = sample.bones.get(bone.name)
        location, rotation, scale = bone.matrix_basis.decompose()
        semantic_id = (
            channel.semantic_id
            if channel is not None
            else previous.semantic_id
            if previous is not None
            else ""
        )
        confidence = (
            min(float(channel.confidence), ik_confidences.get(bone, 1.0))
            if channel is not None
            else ik_confidences.get(
                bone, previous.confidence if previous is not None else 0.0
            )
        )
        sample.bones[bone.name] = _BakedBoneSample(
            bone=bone,
            location=location.copy(),
            rotation=rotation.normalized(),
            scale=scale.copy(),
            key_location=bool(
                bone in ik_location
                or channel is not None
                and channel.location is not None
                or previous is not None
                and previous.key_location
            ),
            key_rotation=bool(
                bone in ik_rotation
                or channel is not None
                and (
                    channel.desired_rotation is not None
                    or channel.direct_basis_rotation is not None
                    or channel.hinge_axis_local is not None
                )
                or previous is not None
                and previous.key_rotation
            ),
            semantic_id=semantic_id,
            confidence=min(max(float(confidence), 0.0), 1.0),
            properties=(dict(previous.properties) if previous is not None else {}),
        )


def _iter_apply_rohm_pose(
    armature,
    entries,
    rig_plan: rig_planning.RigSolvePlan,
    samples: Sequence[_BakedFrameSample],
    pose_global_rotations: Sequence[dict[str, Quaternion]],
) -> Iterator[SequenceApplyProgress]:
    """Rest-align RoHM globals and resolve them through the control rig."""
    if len(pose_global_rotations) != len(samples):
        raise BackendError("RoHM pose corrections do not match the baked sequence")
    rest_basis = rig_plan.rest_basis.to_3x3()
    rest_inverse = rest_basis.inverted_safe()
    temporary = _sequence_state(armature)
    try:
        for index, (sample, corrected_globals) in enumerate(
            zip(samples, pose_global_rotations, strict=True), 1
        ):
            if not corrected_globals:
                yield SequenceApplyProgress(
                    "RoHM: preserving trusted pose", index, len(samples)
                )
                continue
            _scene_set_float(bpy.context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample)
            posed_root_basis = target_basis(
                armature,
                entries,
                rest_basis=rig_plan.rest_basis,
            ).to_3x3()
            target_pose = TargetPose(
                armature,
                None,
                entries,
                rest_basis,
                posed_root_basis,
                rig_plan=rig_plan,
            )
            for semantic_id, anatomical_rotation in corrected_globals.items():
                if semantic_id == "PELVIS":
                    continue
                bone = _mapped_bone(armature, entries, semantic_id)
                source_rotation = sample.rohm_rotations.get(semantic_id)
                if bone is None or source_rotation is None:
                    continue
                if (
                    source_rotation.rotation_difference(anatomical_rotation).angle
                    <= 1.0e-5
                ):
                    continue
                target_delta = (
                    posed_root_basis @ anatomical_rotation.to_matrix() @ rest_inverse
                )
                desired = target_delta @ bone.bone.matrix_local.to_3x3()
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    desired,
                    confidence=0.0,
                    owner="ROHM",
                )
            if target_pose.bones:
                ik_proposals = _prepare_ik_output(target_pose)
                changed_bones, _changed_shapes = _commit_target_pose(
                    target_pose, ik_proposals
                )
                _merge_rohm_pose_into_sample(
                    sample,
                    target_pose,
                    ik_proposals,
                    changed_bones,
                )
            yield SequenceApplyProgress(
                "RoHM: retargeting corrected pose", index, len(samples)
            )
    finally:
        _restore_sequence_state(armature, temporary)


def _iter_rohm_correction(
    armature,
    entries,
    rig_plan: rig_planning.RigSolvePlan,
    samples: list[_BakedFrameSample],
    settings,
    effective_mode: str,
    root_kind: str,
    root_target,
) -> Iterator[SequenceApplyProgress]:
    """Run the explicitly enabled verified RoHM correction backend."""
    from . import rohm_backend, rohm_correction

    correct_pose, correct_root = _rohm_requested_work(samples, settings, effective_mode)
    if not correct_pose and not correct_root:
        return
    runtime = rohm_backend.loaded_session()
    iterator = rohm_correction.iter_correct_samples(
        samples,
        _sequence_timestamps(samples),
        runtime,
        root_motion_mode=effective_mode,
        correct_pose=correct_pose,
        correct_root=correct_root,
        visibility_threshold=float(settings.visibility_threshold),
        seed=10,
    )
    pose_apply_total = len(samples) if correct_pose else 0
    root_apply_total = len(samples) if correct_root else 0
    apply_total = pose_apply_total + root_apply_total
    model_total = 0
    while True:
        try:
            phase, current, total = next(iterator)
        except StopIteration as stopped:
            stats = stopped.value
            break
        model_total = max(model_total, int(total))
        yield SequenceApplyProgress(
            str(phase),
            int(current),
            int(total) + apply_total,
        )
    if correct_pose:
        pose_iterator = _iter_apply_rohm_pose(
            armature,
            entries,
            rig_plan,
            samples,
            stats.pose_global_rotations,
        )
        for progress in pose_iterator:
            yield SequenceApplyProgress(
                progress.phase,
                model_total + progress.current,
                model_total + apply_total,
            )
    if correct_root:
        root_iterator = _iter_apply_rohm_root_deltas(
            armature,
            samples,
            stats.root_deltas,
            root_kind,
            root_target,
        )
        for progress in root_iterator:
            yield SequenceApplyProgress(
                progress.phase,
                model_total + pose_apply_total + progress.current,
                model_total + apply_total,
            )
    _append_bake_warning(
        samples,
        "Experimental RoHM Correction changed "
        f"{stats.corrected_source_frames} source frame(s) at 30 FPS using "
        + " and ".join(
            name
            for enabled, name in (
                (correct_pose, "pose repair"),
                (correct_root, "root correction"),
            )
            if enabled
        ),
    )
    if stats.low_rate_warning:
        _append_bake_warning(
            samples,
            "RoHM upsampled input below 30 FPS; absent motion detail cannot be recovered",
        )
    for warning in stats.warnings:
        _append_bake_warning(samples, f"RoHM quality warning · {warning}")


def _rohm_requested_work(
    samples: Sequence[_BakedFrameSample],
    settings,
    effective_mode: str,
) -> tuple[bool, bool]:
    """Return whether masked pose or root observations justify model inference."""
    if settings is None or not bool(getattr(settings, "enabled", False)):
        return False, False
    threshold = min(
        max(float(getattr(settings, "visibility_threshold", 0.5)), 0.0), 1.0
    )
    pose_missing = any(
        float(sample.rohm_confidence.get(semantic, 0.0)) < threshold
        for sample in samples
        for semantic in motion_filter.SMPLX_SEMANTICS[1:]
    )
    correct_pose = bool(getattr(settings, "learned_pose_residual", False)) and bool(
        pose_missing
    )
    root_values = [float(sample.rohm_root_confidence) for sample in samples]
    root_missing = any(value < threshold for value in root_values)
    root_observed = any(value >= threshold for value in root_values)
    correct_root = bool(
        getattr(settings, "root_trajectory_cleanup", True)
        and effective_mode == "AUTOMATIC"
        and root_missing
        and root_observed
    )
    return correct_pose, correct_root


def _root_baked_channel(sample: _BakedFrameSample):
    return next(
        (
            channel
            for channel in sample.bones.values()
            if channel.semantic_id == "ROOT" and channel.key_location
        ),
        None,
    )


def _root_transform(sample: _BakedFrameSample):
    if sample.root_object is not None:
        return sample.root_object[0].copy(), sample.root_object[1].copy(), "OBJECT"
    channel = _root_baked_channel(sample)
    if channel is not None:
        return channel.location.copy(), channel.rotation.copy(), "BONE"
    return None


def _set_root_transform(
    sample: _BakedFrameSample,
    location: Vector,
    rotation: Quaternion,
) -> bool:
    current = _root_transform(sample)
    if current is None:
        return False
    old_location, _old_rotation, kind = current
    if kind == "OBJECT":
        _location, _rotation, scale = sample.root_object
        sample.root_object = (location.copy(), rotation.copy(), scale.copy())
    else:
        channel = _root_baked_channel(sample)
        channel.location = location.copy()
        channel.rotation = rotation.copy()
    _translate_grounding(sample.grounding, location - old_location)
    return True


def _repair_root_gaps_without_ai(
    samples: Sequence[_BakedFrameSample],
    timestamps: Sequence[float],
) -> int:
    missing = {
        index
        for index, sample in enumerate(samples)
        if float(sample.rohm_root_confidence) <= 0.0
    }
    available = [
        index
        for index, sample in enumerate(samples)
        if index not in missing and _root_transform(sample) is not None
    ]
    if not available:
        return 0
    repaired = 0
    for index in sorted(missing):
        if _root_transform(samples[index]) is None:
            continue
        left = next((value for value in reversed(available) if value < index), None)
        right = next((value for value in available if value > index), None)
        if left is None and right is None:
            continue
        if left is None or right is None:
            source = _root_transform(samples[right if left is None else left])
            location, rotation = source[0], source[1]
        else:
            first = _root_transform(samples[left])
            last = _root_transform(samples[right])
            span = float(timestamps[right]) - float(timestamps[left])
            factor = (
                (float(timestamps[index]) - float(timestamps[left])) / span
                if span > 1.0e-8
                else (index - left) / max(right - left, 1)
            )
            factor = min(max(factor, 0.0), 1.0)
            location = first[0].lerp(last[0], factor)
            rotation = first[1].slerp(last[1], factor).normalized()
        repaired += int(_set_root_transform(samples[index], location, rotation))
    return repaired


def _iter_detection_filter(
    samples: list[_BakedFrameSample],
    settings,
    *,
    rohm_settings=None,
    effective_mode: str = "OFF",
    ui_settings=None,
) -> Iterator[SequenceApplyProgress]:
    """Reject unreliable evidence before RoHM or deterministic local repair."""
    timestamps = _sequence_timestamps(samples)
    yield SequenceApplyProgress("Filtering unreliable detections", 0, 2)
    stats = motion_filter.filter_motion_observations(
        samples,
        timestamps,
        reject_unreliable_limbs=bool(settings.reject_unreliable_limbs),
        reject_unusable_frames=bool(settings.reject_unusable_frames),
        maximum_rejected_fraction=float(settings.maximum_motion_rejected) / 100.0,
    )
    try:
        manual_frames = motion_filter.parse_source_frame_ranges(
            str(getattr(settings, "manual_bad_frames", "")), len(samples)
        )
    except ValueError as exc:
        raise InvalidRequest(f"Additional Bad Source Frames: {exc}") from exc
    manual_rejected = motion_filter.apply_manual_motion_rejections(
        samples,
        manual_frames,
        str(getattr(settings, "manual_bad_scope", "BODY")),
    )
    repaired = motion_filter.repair_body_gaps(samples, timestamps)
    _pose_ai, root_ai = _rohm_requested_work(samples, rohm_settings, effective_mode)
    if not root_ai:
        repaired += _repair_root_gaps_without_ai(samples, timestamps)
    automatic_frames = {frame_index for frame_index, _semantic in stats.rejected}
    automatic_summary = motion_filter.format_source_frame_ranges(automatic_frames)
    manual_summary = motion_filter.format_source_frame_ranges(manual_frames)
    frame_summary = f"Automatic: {automatic_summary} · Manual: {manual_summary}"
    if len(frame_summary) > 240:
        frame_summary = frame_summary[:237].rstrip() + "..."
    if ui_settings is not None:
        with suppress(AttributeError, ReferenceError, RuntimeError):
            ui_settings.filter_rejected_observations = stats.rejected_observations
            ui_settings.filter_rejected_limbs = stats.rejected_limbs
            ui_settings.filter_rejected_frames = stats.rejected_frames
            ui_settings.filter_native_missing = stats.native_missing_observations
            ui_settings.filter_total_observations = stats.total_observations
            ui_settings.filter_manual_frames = len(manual_frames)
            ui_settings.filter_frame_summary = frame_summary
    if stats.rejected_observations:
        _append_bake_warning(
            samples,
            "Detection Filter rejected "
            f"{stats.rejected_observations} available joint sample(s) "
            f"({stats.rejected_frames} frame(s), {stats.rejected_limbs} limb event(s))",
        )
    if manual_frames:
        _append_bake_warning(
            samples,
            "Manual cleanup mask selected "
            f"{len(manual_frames)} source frame(s) and rejected "
            f"{len(manual_rejected)} available joint sample(s)",
        )
    elif str(getattr(settings, "manual_bad_frames", "")).strip():
        _append_bake_warning(
            samples,
            "Manual cleanup mask selected no source frames inside this clip",
        )
    if repaired:
        _append_bake_warning(
            samples,
            f"Detection Filter locally repaired {repaired} missing motion sample(s); "
            "boundary gaps use the nearest reliable pose",
        )
    if stats.observed_fraction < 0.5:
        _append_bake_warning(
            samples,
            "Mostly reconstructed · limited source evidence",
        )
    yield SequenceApplyProgress("Filtering unreliable detections", 2, 2)


def _iter_temporal_correction(
    armature,
    entries,
    samples: list[_BakedFrameSample],
    settings,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_up: Vector,
) -> Iterator[SequenceApplyProgress]:
    if settings.stabilize_hands:
        for sample in samples:
            for channel in sample.bones.values():
                if channel.semantic_id.startswith(
                    (
                        "HAND_",
                        "THUMB_",
                        "INDEX_",
                        "MIDDLE_",
                        "RING_",
                        "PINKY_",
                    )
                ):
                    channel.bone.rotation_mode = "QUATERNION"
    if settings.smooth_motion or settings.stabilize_hands:
        yield from _iter_temporal_channels(armature, samples, settings)
    if settings.ground_contacts and _has_groundable_body_channels(samples):
        yield from _iter_ground_contacts(
            armature,
            entries,
            samples,
            float(settings.smoothing_strength),
            effective_mode,
            root_kind,
            root_target,
            target_up,
        )


def _root_data_paths(root_kind: str, root_target) -> set[str]:
    if root_kind == "BONE":
        return {
            root_target.path_from_id("location"),
            root_target.path_from_id("rotation_euler"),
            root_target.path_from_id("rotation_quaternion"),
            root_target.path_from_id("rotation_axis_angle"),
        }
    return {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle"}


def _key_object_from_stream(
    state: _KeyframeWriteState,
    frame: float,
) -> None:
    obj = state.armature
    obj.keyframe_insert(data_path="location", frame=frame)
    data_path = _rotation_data_path(obj)
    state.previous_object_rotation = _rotation_values_compatible_with(
        obj,
        data_path,
        state.previous_object_rotation,
    )
    obj.keyframe_insert(data_path=data_path, frame=frame)


def _iter_write_baked_sample(
    state: _KeyframeWriteState,
    sample: _BakedFrameSample,
    frame_index: int,
    total_frames: int,
    *,
    phase: str,
) -> Iterator[SequenceApplyProgress]:
    """Write one solved frame in bounded batches and release its caller."""
    state.attach()
    channels = sorted(
        sample.bones.values(),
        key=lambda item: len(item.bone.parent_recursive),
    )
    batch_size = 8
    for offset in range(0, len(channels), batch_size):
        with linear_keyframe_defaults():
            for channel in channels[offset : offset + batch_size]:
                channel.bone.matrix_basis = Matrix.LocRotScale(
                    channel.location,
                    channel.rotation,
                    channel.scale,
                )
                if channel.key_location or channel.key_rotation:
                    _key_bone_from_sample(
                        channel.bone,
                        sample.destination,
                        location=channel.key_location,
                        rotation=channel.key_rotation,
                        previous_rotations=state.previous_rotations,
                    )
                    state.total_bones += 1
                for property_name, value in channel.properties.items():
                    channel.bone[property_name] = value
                    channel.bone.keyframe_insert(
                        data_path=f'["{property_name}"]',
                        frame=sample.destination,
                        group=channel.bone.name,
                    )
        state.detach()
        yield SequenceApplyProgress(phase, frame_index - 1, total_frames)
        state.attach()

    if sample.root_object is not None:
        location, rotation, scale = sample.root_object
        state.armature.matrix_basis = Matrix.LocRotScale(
            location,
            rotation,
            scale,
        )
        with linear_keyframe_defaults():
            _key_object_from_stream(state, sample.destination)
        state.total_bones += 1

    shapes = list(sample.shapes.values())
    for offset in range(0, len(shapes), batch_size):
        with linear_keyframe_defaults():
            for shape in shapes[offset : offset + batch_size]:
                shape.block.value = shape.value
                _key_shape(
                    shape.block,
                    sample.destination,
                    set_interpolation=False,
                )
                state.total_shapes += 1
        state.detach()
        yield SequenceApplyProgress(phase, frame_index - 1, total_frames)
        state.attach()

    state.detach()
    yield SequenceApplyProgress(phase, frame_index, total_frames)


def _finish_keyframe_stream(
    state: _KeyframeWriteState,
    final_frame: float,
) -> tuple[int, int]:
    state.attach()
    _scene_set_float(bpy.context.scene, final_frame)
    state.armature.update_tag(refresh={"OBJECT"})
    bpy.context.view_layer.update()
    return state.total_bones, state.total_shapes


def _iter_commit_baked_samples(
    armature,
    samples: list[_BakedFrameSample],
    root_kind: str,
    root_target,
    root_enabled: bool,
) -> Iterator[SequenceApplyProgress]:
    action = _ensure_action(armature)
    start_frame = samples[0].destination
    end_frame = samples[-1].destination
    replacement_paths: set[str] = set()
    for sample in samples:
        for channel in sample.bones.values():
            if channel.key_location:
                replacement_paths.add(channel.bone.path_from_id("location"))
            if channel.key_rotation:
                replacement_paths.update(
                    {
                        channel.bone.path_from_id("rotation_euler"),
                        channel.bone.path_from_id("rotation_quaternion"),
                        channel.bone.path_from_id("rotation_axis_angle"),
                    }
                )
            replacement_paths.update(
                channel.bone.path_from_id(f'["{property_name}"]')
                for property_name in channel.properties
            )
    if root_enabled:
        replacement_paths.update(_root_data_paths(root_kind, root_target))
    _remove_keys_in_interval(action, replacement_paths, start_frame, end_frame)

    shape_paths_by_data: dict[object, set[str]] = {}
    for sample in samples:
        for shape in sample.shapes.values():
            shape_paths_by_data.setdefault(shape.block.id_data, set()).add(
                shape.block.path_from_id("value")
            )
    for shape_data, shape_paths in shape_paths_by_data.items():
        animation = getattr(shape_data, "animation_data", None)
        if animation is not None and animation.action is not None:
            _remove_keys_in_interval(
                animation.action,
                shape_paths,
                start_frame,
                end_frame,
            )
    state = _KeyframeWriteState(armature)
    total_frames = len(samples)
    for index, sample in enumerate(samples, 1):
        yield from _iter_write_baked_sample(
            state,
            sample,
            index,
            total_frames,
            phase="Writing keyframes",
        )
    for curve in _action_fcurves(action):
        if curve.data_path not in replacement_paths:
            continue
        for point in curve.keyframe_points:
            if start_frame - 1.0e-5 <= float(point.co.x) <= end_frame + 1.0e-5:
                point.interpolation = "LINEAR"
        curve.update()
    for shape_data, shape_paths in shape_paths_by_data.items():
        animation = getattr(shape_data, "animation_data", None)
        if animation is None or animation.action is None:
            continue
        for curve in _action_fcurves(animation.action):
            if curve.data_path not in shape_paths:
                continue
            for point in curve.keyframe_points:
                if start_frame - 1.0e-5 <= float(point.co.x) <= end_frame + 1.0e-5:
                    point.interpolation = "LINEAR"
            curve.update()
    return _finish_keyframe_stream(state, samples[-1].destination)


def _commit_baked_samples(
    armature,
    samples: list[_BakedFrameSample],
    root_kind: str,
    root_target,
    root_enabled: bool,
) -> tuple[int, int]:
    return _drain(
        _iter_commit_baked_samples(
            armature, samples, root_kind, root_target, root_enabled
        )
    )


def _iter_prepare_camera_motion_track(
    request,
    inputs: Sequence[camera_motion_correction.CameraMotionInput],
) -> Iterator[SequenceApplyProgress]:
    """Load or compute the derived RGB camera track before animation writes."""
    source_identity = dict(getattr(request, "source_identity", {}) or {})
    expected_key = camera_motion_correction.cache_key(source_identity, inputs)
    directory = detection_cache.camera_motion_directory(create_root=False)
    from .dependency_bootstrap import import_camera_dependencies

    try:
        cv2, np = import_camera_dependencies()
    except ImportError as exc:
        return None, "Unavailable", str(exc)
    if bool(getattr(request, "keep_detection_cache", False)):
        cached = camera_motion_correction.load_sidecar(directory, expected_key, np)
        if cached is not None:
            yield SequenceApplyProgress("Camera Motion: loaded cached track", 1, 1)
            return cached, "Ready", ""

    source_path = detection_cache.existing_media_path(
        str(getattr(request, "analysis_path", "") or "")
    )
    if not source_path:
        return None, "Not Prepared", ""
    estimator = camera_motion_correction.iter_estimate_track(
        source_path,
        inputs,
        cv2,
        np,
    )
    try:
        while True:
            try:
                phase, current, total = next(estimator)
            except StopIteration as stopped:
                track = stopped.value
                break
            yield SequenceApplyProgress(str(phase), int(current), int(total))
    finally:
        estimator.close()

    status = "Computed"
    warning = ""
    if bool(getattr(request, "keep_detection_cache", False)):
        try:
            directory = detection_cache.camera_motion_directory(create_root=True)
            camera_motion_correction.store_sidecar(
                directory,
                expected_key,
                track,
                np,
            )
            limit = max(0, int(getattr(request, "cache_limit_bytes", 0)))
            if limit and detection_cache.cache_usage_bytes() > limit:
                detection_cache.clear_camera_motion_cache()
                status = "Computed (not cached)"
                warning = (
                    "Camera track was used but not retained because the cache limit "
                    "would be exceeded"
                )
            else:
                status = "Ready"
        except (OSError, RuntimeError, ValueError) as exc:
            status = "Computed (not cached)"
            warning = f"Camera track was used but could not be cached: {exc}"
    return track, status, warning


def iter_apply_sequence(
    armature,
    frames: Iterable[CanonicalFrame],
    settings,
    start_frame: float,
) -> Iterator[SequenceApplyProgress]:
    """Cooperatively solve and key a sequence without long UI stalls.

    Blender data is deliberately touched only by the caller's main thread.
    Yielding between cache reads, solved frames, cleanup channels, and written
    frames lets a modal operator return control to Blender regularly.
    """
    total_hint = len(frames) if isinstance(frames, Sequence) else 0
    scene = bpy.context.scene
    scene_fps = scene.render.fps / scene.render.fps_base
    state = _sequence_state(armature)
    samples: list[_BakedFrameSample] = []
    warning_sink: list[str] = []
    config = armature.mtp_rig_config
    postprocess_config = getattr(
        settings, "postprocess_config", None
    ) or compile_postprocess_config(config)
    temporal_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "TEMPORAL_CORRECTION"
        ),
        None,
    )
    rohm_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "ROHM_CORRECTION"
        ),
        None,
    )
    capture_rohm_input = bool(
        rohm_settings
        and rohm_settings.enabled
        and (
            bool(getattr(rohm_settings, "learned_pose_residual", False))
            or (
                bool(getattr(rohm_settings, "root_trajectory_cleanup", True))
                and getattr(settings, "root_motion_mode", "AUTOMATIC") == "AUTOMATIC"
            )
        )
    )
    detection_filter_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "DETECTION_FILTER"
        ),
        None,
    )
    camera_motion_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "CAMERA_MOTION_CORRECTION"
        ),
        None,
    )
    camera_motion_ui_settings = next(
        (
            item
            for item in config.post_processing_modules
            if item.module_id == "CAMERA_MOTION_CORRECTION"
        ),
        None,
    )
    detection_filter_ui_settings = next(
        (
            item
            for item in config.post_processing_modules
            if item.module_id == "DETECTION_FILTER"
        ),
        None,
    )
    entries = getattr(settings, "mapping_entries", None) or mapping.mapping_by_id(
        config
    )
    rig_plan = rig_planning.compile_rig_solve_plan(
        armature,
        entries,
        lambda: target_rest_basis(armature, entries),
    )
    requested_mode = getattr(settings, "root_motion_mode", "AUTOMATIC")
    root_kind, root_target = _sequence_root_target(armature, entries)
    root_enabled = requested_mode != "OFF"
    offline_processing = bool(
        (
            camera_motion_settings
            and camera_motion_settings.enabled
            and requested_mode == "AUTOMATIC"
            and str(getattr(settings, "media_kind", "")) == "VIDEO"
        )
        or (detection_filter_settings and detection_filter_settings.enabled)
        or (rohm_settings and rohm_settings.enabled)
        or (
            temporal_settings
            and temporal_settings.enabled
            and (
                temporal_settings.smooth_motion
                or temporal_settings.ground_contacts
                or temporal_settings.stabilize_hands
            )
        )
    )
    frame_spool = None
    root_samples: list[_RootMotionSample] = []
    planned_deltas: list[Vector] = []
    planned_yaws: list[float] = []
    effective_mode = requested_mode
    frame_count = 0
    previous_timestamp_ms = None
    camera_inputs: list[camera_motion_correction.CameraMotionInput] = []
    camera_track: camera_motion_correction.CameraMotionTrack | None = None
    camera_track_status = "Not Prepared"
    camera_preflight_warning = ""
    target_lateral = Vector((1.0, 0.0, 0.0))
    object_delta_basis = Matrix.Identity(3)
    source_to_target_scale = 1.0

    if requested_mode == "AUTOMATIC":
        # The conditional spool is closed by the encompassing generator
        # ``finally`` block, including GeneratorExit during cancellation.
        frame_spool = FrameSpool()
        try:
            for detected in frames:
                timestamp_ms = int(detected.timestamp_ms)
                if (
                    previous_timestamp_ms is not None
                    and timestamp_ms <= previous_timestamp_ms
                ):
                    raise MediaError("Detected video timestamps must increase")
                previous_timestamp_ms = timestamp_ms
                frame_spool.append(detected)
                root_samples.append(_root_motion_sample(detected))
                camera_inputs.append(
                    camera_motion_correction.input_from_frame(detected)
                )
                frame_count += 1
                yield SequenceApplyProgress(
                    "Loading detections",
                    frame_count,
                    total_hint,
                )
            if not frame_count:
                raise MediaError("No detectable frames were produced")
        except BaseException:
            frame_spool.close()
            raise

        frame_iterator = iter(frame_spool)

        if (
            camera_motion_settings
            and camera_motion_settings.enabled
            and str(getattr(settings, "media_kind", "")) == "VIDEO"
        ):
            try:
                camera_preflight = _iter_prepare_camera_motion_track(
                    settings,
                    camera_inputs,
                )
                while True:
                    try:
                        yield next(camera_preflight)
                    except StopIteration as stopped:
                        (
                            camera_track,
                            camera_track_status,
                            camera_preflight_warning,
                        ) = stopped.value
                        break
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                camera_track = None
                camera_track_status = "Error"
                camera_preflight_warning = f"Camera Motion Correction skipped: {exc}"
            if camera_motion_ui_settings is not None:
                with suppress(AttributeError, ReferenceError, RuntimeError):
                    camera_motion_ui_settings.camera_track_status = camera_track_status
        total_frames = frame_count
    else:
        frame_iterator = iter(frames)
        total_frames = total_hint
        frame_count = 0
        previous_timestamp_ms = None

    keyframe_state = None
    last_destination = float(start_frame)
    try:
        processing_session = PostProcessingSession()
        target_up = None
        baseline = None
        for index, detected in enumerate(frame_iterator, 1):
            timestamp_ms = int(detected.timestamp_ms)
            if requested_mode != "AUTOMATIC":
                if (
                    previous_timestamp_ms is not None
                    and timestamp_ms <= previous_timestamp_ms
                ):
                    raise MediaError("Detected video timestamps must increase")
                previous_timestamp_ms = timestamp_ms
                frame_count = index
                yield SequenceApplyProgress(
                    "Loading detections",
                    index,
                    total_frames,
                )
            destination = float(start_frame) + timestamp_ms * scene_fps / 1000.0
            last_destination = destination
            _scene_set_float(scene, destination)
            if index == 1:
                prepare_limb_ownership(armature)
                if root_enabled and not bool(
                    getattr(settings, "keep_original_root_position", False)
                ):
                    _reset_root_location(root_target)
                target_space = target_basis(armature, entries)
                target_up = target_space @ Vector((0.0, 0.0, 1.0))
                baseline = (
                    root_target.matrix.copy()
                    if root_kind == "BONE"
                    else root_target.matrix_world.copy()
                )
                if requested_mode == "AUTOMATIC":
                    (
                        planned_deltas,
                        planned_yaws,
                        effective_mode,
                        root_warnings,
                    ) = _root_motion_deltas(
                        armature,
                        entries,
                        root_samples,
                        requested_mode,
                    )
                    valid_root_samples = [
                        sample
                        for sample in root_samples
                        if sample.observation is not None and sample.observation.valid
                    ]
                    if valid_root_samples:
                        reference = valid_root_samples[0]
                        lateral = target_space @ (
                            reference.source_basis.inverted_safe()
                            @ Vector((1.0, 0.0, 0.0))
                        )
                        normalized_up = target_up.normalized()
                        lateral -= normalized_up * lateral.dot(normalized_up)
                        if lateral.length > 1.0e-8:
                            target_lateral = lateral.normalized()
                        target_height = _target_body_height(armature, entries)
                        source_to_target_scale = (
                            target_height / reference.source_height
                            if target_height > 1.0e-6
                            and reference.source_height > 1.0e-6
                            else 1.0
                        )
                        source_to_target_scale = min(
                            max(source_to_target_scale, 0.1), 10.0
                        )
                    object_delta_basis = baseline.to_3x3()
                    for warning in root_warnings:
                        if warning not in warning_sink:
                            warning_sink.append(warning)
                        if warning not in detected.warnings:
                            detected.warnings.append(warning)
                    if camera_preflight_warning:
                        if camera_preflight_warning not in warning_sink:
                            warning_sink.append(camera_preflight_warning)
                        if camera_preflight_warning not in detected.warnings:
                            detected.warnings.append(camera_preflight_warning)
                _ensure_action(armature)
                keyframe_state = _KeyframeWriteState(armature)

            delta = (
                planned_deltas[index - 1] if effective_mode == "AUTOMATIC" else Vector()
            )
            yaw = planned_yaws[index - 1] if effective_mode == "AUTOMATIC" else 0.0
            if root_enabled:
                _set_root_pose(
                    root_kind,
                    root_target,
                    baseline,
                    delta,
                    yaw,
                    target_up,
                )
            frame_solver = _iter_capture_baked_frame(
                armature,
                detected,
                destination,
                settings,
                processing_session,
                root_kind,
                root_target,
                root_enabled,
                postprocess_config,
                rig_plan,
                warning_sink,
                capture_rohm_input,
            )
            while True:
                try:
                    solve_phase, solve_current, solve_total = next(frame_solver)
                except StopIteration as stopped:
                    sample = stopped.value
                    break
                yield SequenceApplyProgress(
                    solve_phase,
                    (index - 1) * solve_total + solve_current,
                    total_frames * solve_total,
                )
            if offline_processing:
                samples.append(sample)
            yield SequenceApplyProgress(
                "Solving animation",
                index,
                total_frames,
            )
            yield from _iter_write_baked_sample(
                keyframe_state,
                sample,
                index,
                total_frames,
                phase="Streaming keyframes",
            )
        if not frame_count:
            raise MediaError("No detectable frames were produced")
    finally:
        _restore_sequence_state(armature, state)
        if frame_spool is not None:
            frame_spool.close()

    if offline_processing:
        context = SequenceProcessingContext(
            samples=samples,
            timestamps=_sequence_timestamps(samples),
            root_motion_mode=effective_mode,
            process_camera_motion=lambda module_settings: _iter_camera_motion_correction(
                armature,
                samples,
                module_settings,
                effective_mode,
                root_kind,
                root_target,
                target_lateral,
                object_delta_basis,
                source_to_target_scale,
                camera_track,
                camera_motion_ui_settings,
                camera_track_status,
                str(getattr(settings, "media_kind", "")) == "VIDEO",
            ),
            process_temporal=lambda module_settings: _iter_temporal_correction(
                armature,
                entries,
                samples,
                module_settings,
                effective_mode,
                root_kind,
                root_target,
                target_up,
            ),
            process_detection_filter=lambda module_settings: _iter_detection_filter(
                samples,
                module_settings,
                rohm_settings=rohm_settings,
                effective_mode=effective_mode,
                ui_settings=detection_filter_ui_settings,
            ),
            process_rohm=lambda module_settings: _iter_rohm_correction(
                armature,
                entries,
                rig_plan,
                samples,
                module_settings,
                effective_mode,
                root_kind,
                root_target,
            ),
        )
        yield from iter_sequence_pipeline(context, postprocess_config)
        keyframe_state.attach()
        total_bones, total_shapes = yield from _iter_commit_baked_samples(
            armature,
            samples,
            root_kind,
            root_target,
            root_enabled,
        )
    else:
        total_bones, total_shapes = _finish_keyframe_stream(
            keyframe_state,
            last_destination,
        )
    return frame_count, total_bones, total_shapes, tuple(warning_sink)


def apply_sequence(
    armature,
    frames: Iterable[CanonicalFrame],
    settings,
    start_frame: float,
) -> tuple[int, int, int, tuple[str, ...]]:
    """Synchronous API retained for scripts, tests, and single-shot callers."""
    return _drain(iter_apply_sequence(armature, frames, settings, start_frame))
