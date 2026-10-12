"""Mesh-free semantic-joint adapter for the optional RoHM ONNX runtime."""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .backends.rohm.runtime import (
    BETAS_SLICE,
    FEATURE_COUNT,
    MODEL_FPS,
    MODEL_WINDOW_FRAMES,
    POSE_COUNT,
    SOURCE_WINDOW_FRAMES,
    TRAJECTORY_ABSOLUTE_INDICES,
    TRAJECTORY_COUNT,
    visibility_vector,
)
from .core.errors import BackendError
from .geometry import Matrix, Quaternion, Vector

SMPLX_SEMANTICS = (
    "PELVIS",
    "UPPER_LEG_L",
    "UPPER_LEG_R",
    "SPINE_01",
    "LOWER_LEG_L",
    "LOWER_LEG_R",
    "SPINE_02",
    "FOOT_L",
    "FOOT_R",
    "CHEST",
    "TOE_L",
    "TOE_R",
    "NECK",
    "SHOULDER_L",
    "SHOULDER_R",
    "HEAD",
    "UPPER_ARM_L",
    "UPPER_ARM_R",
    "LOWER_ARM_L",
    "LOWER_ARM_R",
    "HAND_L",
    "HAND_R",
)

# Official SMPL-X body-joint kinematic tree for the 22 joints above.  PoseNet
# predicts local rotations for joints 1..21; those rotations are meaningful
# only in this hierarchy and must never be treated as target-rig bone locals.
SMPLX_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
)

_LEFT_FOOT = (7, 10)
_RIGHT_FOOT = (8, 11)
_FACE_JOINTS = (2, 1, 17, 16)  # right hip, left hip, right/left shoulder
_IDENTITY = np.eye(3, dtype=np.float64)

# The detection filter has already replaced untrusted samples with finite
# quaternion interpolation (or a nearest trusted boundary pose).  The learned
# result is therefore a residual proposal, not permission to replace that safe
# baseline wholesale.  The SAM3 validation found two degrees to be the least
# harmful non-zero bound; larger bounds regressed the repaired sequence and
# allowed collapses approaching 180 degrees.
MAX_LEARNED_GLOBAL_POSE_DELTA_DEGREES = 2.0


class RohmCorrectionError(BackendError):
    """A semantic conversion or corrected-motion validation failure."""


@dataclass(frozen=True)
class RepresentationContext:
    scale: float
    floor: float
    root_xy: np.ndarray
    canonical_rotation: np.ndarray


@dataclass(frozen=True)
class CorrectionResult:
    representation: np.ndarray
    joints: np.ndarray
    root_applied: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrectionStats:
    source_fps: float
    model_frames: int
    windows: int
    corrected_source_frames: int
    corrected_model_frames: int
    low_rate_warning: bool
    root_deltas: tuple[tuple[float, float, float], ...]
    observed_fraction: float = 1.0
    warnings: tuple[str, ...] = ()
    pose_global_rotations: tuple[dict[str, Quaternion], ...] = ()


def matrix_to_rotation_6d(value: np.ndarray) -> np.ndarray:
    """Encode RoHM's column-major Zhou 6D rotation representation."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise RohmCorrectionError("invalid rotation matrix")
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,)).astype(np.float32)


def rotation_6d_to_matrix(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    if data.shape[-1] != 6 or not np.isfinite(data).all():
        raise RohmCorrectionError("invalid RoHM 6D rotation")
    columns = data.reshape(data.shape[:-1] + (3, 2))
    first = columns[..., :, 0]
    second = columns[..., :, 1]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1.0e-8):
        raise RohmCorrectionError("degenerate RoHM rotation axis")
    b1 = first / first_norm
    second = second - np.sum(b1 * second, axis=-1, keepdims=True) * b1
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1.0e-8):
        raise RohmCorrectionError("collinear RoHM rotation axes")
    b2 = second / second_norm
    b3 = np.cross(b1, b2)
    result = np.stack((b1, b2, b3), axis=-1)
    return result.astype(np.float32)


def global_anatomical_to_local(global_rotations: np.ndarray) -> np.ndarray:
    """Convert global anatomical deltas to SMPL-X hierarchy-local rotations."""
    value = np.asarray(global_rotations, dtype=np.float64)
    if value.ndim != 4 or value.shape[1:] != (22, 3, 3):
        raise RohmCorrectionError(
            "global anatomical rotations must have shape [T, 22, 3, 3]"
        )
    if not np.isfinite(value).all():
        raise RohmCorrectionError("global anatomical rotations must be finite")
    local = np.empty((len(value), 21, 3, 3), dtype=np.float64)
    for joint in range(1, len(SMPLX_SEMANTICS)):
        parent = SMPLX_PARENTS[joint]
        local[:, joint - 1] = np.swapaxes(value[:, parent], -1, -2) @ value[:, joint]
    return local.astype(np.float32)


def local_to_global_anatomical(local_rotations: np.ndarray) -> np.ndarray:
    """Expand SMPL-X local rotations into global, rest-relative transforms."""
    value = np.asarray(local_rotations, dtype=np.float64)
    if value.ndim != 4 or value.shape[1:] != (21, 3, 3):
        raise RohmCorrectionError(
            "SMPL-X local rotations must have shape [T, 21, 3, 3]"
        )
    if not np.isfinite(value).all():
        raise RohmCorrectionError("SMPL-X local rotations must be finite")
    result = np.zeros((len(value), 22, 3, 3), dtype=np.float64)
    result[:, 0] = _IDENTITY
    for joint in range(1, len(SMPLX_SEMANTICS)):
        parent = SMPLX_PARENTS[joint]
        result[:, joint] = result[:, parent] @ value[:, joint - 1]
    return result.astype(np.float32)


def _rotation_z(angle: np.ndarray | float) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float64)
    cosine, sine = np.cos(value), np.sin(value)
    result = np.zeros(value.shape + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1.0
    return result


def _safe_unit(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(value, axis=-1, keepdims=True)
    result = np.divide(value, lengths, out=np.zeros_like(value), where=lengths > 1e-8)
    invalid = lengths[..., 0] <= 1e-8
    if np.any(invalid):
        result[invalid] = fallback
    return result


def _canonicalize(
    joints: np.ndarray, body_height: float
) -> tuple[np.ndarray, RepresentationContext]:
    points = np.asarray(joints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (22, 3):
        raise RohmCorrectionError("RoHM joints must have shape [T, 22, 3]")
    if len(points) < 2 or not np.isfinite(points).all():
        raise RohmCorrectionError("RoHM requires at least two finite joint frames")
    if not math.isfinite(body_height) or body_height <= 1.0e-5:
        raise RohmCorrectionError("target-rig body height is invalid")
    # RoHM was trained in metric space. Normalize arbitrary target rigs to a
    # neutral 1.70 m stature while preserving the target's rest proportions.
    scale = 1.70 / float(body_height)
    scaled = points * scale
    floor = float(np.min(scaled[..., 2]))
    scaled[..., 2] -= floor
    root_xy = scaled[0, 0] * np.asarray((1.0, 1.0, 0.0))
    scaled -= root_xy
    right_hip, left_hip, right_shoulder, left_shoulder = _FACE_JOINTS
    across = (
        scaled[0, right_hip]
        - scaled[0, left_hip]
        + scaled[0, right_shoulder]
        - scaled[0, left_shoulder]
    )
    across[2] = 0.0
    across = _safe_unit(across[None], np.asarray((1.0, 0.0, 0.0)))[0]
    forward = _safe_unit(
        np.cross(np.asarray((0.0, 0.0, 1.0)), across)[None],
        np.asarray((0.0, 1.0, 0.0)),
    )[0]
    x_axis = across
    y_axis = forward
    canonical_rotation = np.stack((x_axis, y_axis, (0.0, 0.0, 1.0)), axis=1)
    canonical = scaled @ canonical_rotation
    return canonical, RepresentationContext(
        scale=scale,
        floor=floor,
        root_xy=root_xy.astype(np.float64),
        canonical_rotation=canonical_rotation.astype(np.float64),
    )


def _facing_angles(joints: np.ndarray) -> np.ndarray:
    # Preserve upstream's intentionally asymmetric unpacking: the canonical
    # transform calls indices 2/1 right/left, while representation extraction
    # assigns the same tuple to left/right before forming the across vector.
    left_hip, right_hip, right_shoulder, left_shoulder = _FACE_JOINTS
    across = (
        joints[:, right_hip]
        - joints[:, left_hip]
        + joints[:, right_shoulder]
        - joints[:, left_shoulder]
    )
    across[:, 2] = 0.0
    across = _safe_unit(across, np.asarray((1.0, 0.0, 0.0)))
    forward = _safe_unit(
        np.cross(np.asarray((0.0, 0.0, 1.0)), across),
        np.asarray((0.0, 1.0, 0.0)),
    )
    # Rotation taking the current forward vector onto +Y.
    theta = np.arctan2(forward[:, 0], forward[:, 1])
    theta = np.unwrap(theta)
    theta[0] = 0.0
    return theta


def _angular_velocity(rotations: np.ndarray) -> np.ndarray:
    delta = rotations[1:] - rotations[:-1]
    tensor = delta @ np.swapaxes(rotations[:-1], -1, -2)
    return np.stack(
        (
            (-tensor[:, 1, 2] + tensor[:, 2, 1]) * 0.5,
            (tensor[:, 0, 2] - tensor[:, 2, 0]) * 0.5,
            (-tensor[:, 0, 1] + tensor[:, 1, 0]) * 0.5,
        ),
        axis=-1,
    )


def _foot_contacts(joints: np.ndarray, threshold: float = 5.0e-5) -> np.ndarray:
    velocity_sq = np.sum(np.diff(joints, axis=0) ** 2, axis=-1)
    left = (velocity_sq[:, _LEFT_FOOT] < threshold) & (
        joints[:-1, _LEFT_FOOT, 2] < np.asarray((0.18, 0.15))
    )
    right = (velocity_sq[:, _RIGHT_FOOT] < threshold) & (
        joints[:-1, _RIGHT_FOOT, 2] < np.asarray((0.18, 0.15))
    )
    return np.concatenate((left, right), axis=-1).astype(np.float32)


def build_representation(
    joints: np.ndarray,
    body_rotations: np.ndarray,
    *,
    body_height: float,
    global_rotations: np.ndarray | None = None,
) -> tuple[np.ndarray, RepresentationContext]:
    """Build RoHM's 294-channel, 30 FPS representation without a body model."""
    canonical, context = _canonicalize(joints, body_height)
    frame_count = len(canonical)
    local = np.asarray(body_rotations, dtype=np.float64)
    if local.shape == (frame_count, 22, 3, 3):
        local = local[:, 1:]
    if local.shape != (frame_count, 21, 3, 3) or not np.isfinite(local).all():
        raise RohmCorrectionError("body rotations must have shape [T, 21, 3, 3]")
    theta = _facing_angles(canonical)
    root_rotation = _rotation_z(theta)
    local_positions = canonical.copy()
    local_positions[..., :2] -= canonical[:, None, 0, :2]
    local_positions = np.einsum("tij,tkj->tki", root_rotation, local_positions)
    global_velocity = np.diff(canonical, axis=0)
    local_velocity = np.einsum("tij,tkj->tki", root_rotation[:-1], global_velocity)
    root_velocity = np.einsum(
        "tij,tj->ti", root_rotation[1:], np.diff(canonical[:, 0], axis=0)
    )
    if global_rotations is None:
        global_matrix = _rotation_z(-theta)
    else:
        source_global = np.asarray(global_rotations, dtype=np.float64)
        if source_global.shape != (frame_count, 3, 3):
            raise RohmCorrectionError("global rotations must have shape [T, 3, 3]")
        global_matrix = np.einsum(
            "ij,tjk->tik", context.canonical_rotation.T, source_global
        )
    root_angle = theta * 0.5
    angle_velocity = np.diff(root_angle)
    result = np.zeros((frame_count - 1, FEATURE_COUNT), dtype=np.float32)
    result[:, 0] = root_angle[:-1]
    result[:, 1] = angle_velocity
    result[:, 2:4] = canonical[:-1, 0, :2]
    result[:, 4:6] = root_velocity[:, :2]
    result[:, 6] = canonical[:-1, 0, 2]
    result[:, 7:13] = matrix_to_rotation_6d(global_matrix[:-1])
    result[:, 13:16] = _angular_velocity(global_matrix)
    result[:, 16:19] = canonical[:-1, 0]
    result[:, 19:22] = np.diff(canonical[:, 0], axis=0)
    result[:, 22:88] = local_positions[:-1].reshape(frame_count - 1, -1)
    result[:, 88:154] = local_velocity.reshape(frame_count - 1, -1)
    result[:, 154:280] = matrix_to_rotation_6d(local[:-1]).reshape(frame_count - 1, -1)
    result[:, BETAS_SLICE] = 0.0
    result[:, 290:294] = _foot_contacts(canonical)
    if not np.isfinite(result).all():
        raise RohmCorrectionError("RoHM representation contains non-finite values")
    return result, context


def recover_joints(
    representation: np.ndarray, context: RepresentationContext | None = None
) -> np.ndarray:
    """Recover 22 semantic joints using only the joint trajectory channels."""
    value = np.asarray(representation, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != FEATURE_COUNT:
        raise RohmCorrectionError("RoHM representation must have shape [T, 294]")
    angle = value[:, 0] * 2.0
    inverse_root = _rotation_z(-angle)
    local = value[:, 22:88].reshape(-1, 22, 3)
    joints = np.einsum("tij,tkj->tki", inverse_root, local)
    joints[..., 0] += value[:, None, 2]
    joints[..., 1] += value[:, None, 3]
    joints[:, 0] = np.stack((value[:, 2], value[:, 3], value[:, 6]), axis=-1)
    if context is not None:
        joints = joints @ context.canonical_rotation.T
        joints += context.root_xy
        joints[..., 2] += context.floor
        joints /= context.scale
    if not np.isfinite(joints).all():
        raise RohmCorrectionError("RoHM recovered non-finite joints")
    return joints.astype(np.float32)


def _rebuild_trajectory(value: np.ndarray) -> np.ndarray:
    """Recompute RoHM's relative trajectory channels from absolute output."""
    result = np.asarray(value, dtype=np.float32).copy()
    angle = np.unwrap(result[:, 0].astype(np.float64))
    result[:, 1] = np.diff(angle, append=angle[-1]).astype(np.float32)
    root = np.stack((result[:, 2], result[:, 3], result[:, 6]), axis=-1)
    root_delta = np.diff(root, axis=0, append=root[-1:])
    rotations = _rotation_z(result[:, 0].astype(np.float64) * 2.0)
    root_velocity = np.einsum("tij,tj->ti", rotations, root_delta)
    result[:, 4:6] = root_velocity[:, :2]
    global_matrix = rotation_6d_to_matrix(result[:, 7:13])
    angular = _angular_velocity(global_matrix)
    result[:-1, 13:16] = angular
    result[-1, 13:16] = angular[-1] if len(angular) else 0.0
    result[:-1, 19:22] = np.diff(result[:, 16:19], axis=0)
    result[-1, 19:22] = 0.0
    return result


def _replace_absolute_trajectory(
    normalized: np.ndarray, trajectory: np.ndarray, runtime
) -> np.ndarray:
    result = np.asarray(normalized, dtype=np.float32).copy()
    result[:, TRAJECTORY_ABSOLUTE_INDICES] = trajectory[0]
    denormalized = _rebuild_trajectory(runtime.denormalize(result))
    rebuilt = runtime.normalize(denormalized)
    result[:, :TRAJECTORY_COUNT] = rebuilt[:, :TRAJECTORY_COUNT]
    return result


def _validate_corrected(
    source: np.ndarray,
    corrected: np.ndarray,
    *,
    max_root_step: float = 0.50,
    max_root_correction: float = 2.00,
) -> tuple[str, ...]:
    if corrected.shape != source.shape or not np.isfinite(corrected).all():
        raise RohmCorrectionError("RoHM returned an invalid corrected representation")
    warnings: list[str] = []
    source_root = source[:, 2:4]
    corrected_root = corrected[:, 2:4]
    largest_step = (
        np.max(np.linalg.norm(np.diff(corrected_root, axis=0), axis=-1))
        if len(corrected_root) > 1
        else 0.0
    )
    if largest_step > max_root_step:
        warnings.append("RoHM generated an unusually large root step")
    largest_correction = np.max(np.linalg.norm(corrected_root - source_root, axis=-1))
    if largest_correction > max_root_correction:
        warnings.append("RoHM generated an unusually large root correction")
    rotations = rotation_6d_to_matrix(corrected[:, 154:280].reshape(-1, 6))
    determinant = np.linalg.det(rotations)
    if np.max(np.abs(determinant - 1.0)) > 1.0e-4:
        raise RohmCorrectionError("RoHM produced an invalid body rotation")
    source_pose = source[:, 22:88].reshape(-1, 22, 3)
    corrected_pose = corrected[:, 22:88].reshape(-1, 22, 3)
    source_extent = float(
        np.median(np.linalg.norm(source_pose - source_pose[:, :1], axis=-1))
    )
    corrected_extent = float(
        np.median(np.linalg.norm(corrected_pose - corrected_pose[:, :1], axis=-1))
    )
    if source_extent > 1.0e-5 and corrected_extent < source_extent * 0.25:
        warnings.append("RoHM generated motion with a possibly collapsed body pose")
    return tuple(warnings)


_WINDOW_INFERENCE_UNITS = 2200


def _iter_runtime_pass(
    runtime,
    cooperative_name: str,
    blocking_name: str,
    args: tuple,
    kwargs: dict,
    phase: str,
    offset: int,
    units: int,
) -> Iterator[tuple[str, int, int]]:
    cooperative = getattr(runtime, cooperative_name, None)
    if cooperative is None:
        result = getattr(runtime, blocking_name)(*args, **kwargs)
        yield phase, offset + units, _WINDOW_INFERENCE_UNITS
        return result
    iterator = cooperative(*args, **kwargs)
    while True:
        try:
            current, total = next(iterator)
        except StopIteration as stopped:
            return stopped.value
        ratio = float(current) / max(float(total), 1.0)
        yield (
            phase,
            offset + min(max(int(round(units * ratio)), 0), units),
            _WINDOW_INFERENCE_UNITS,
        )


def iter_correct_window(
    representation: np.ndarray,
    joint_visibility: np.ndarray,
    runtime,
    *,
    root_motion_mode: str,
    correct_pose: bool = True,
    correct_root: bool = True,
    seed: int = 10,
    root_observed: np.ndarray | None = None,
) -> Iterator[tuple[str, int, int]]:
    """Run one RoHM window while yielding after every ONNX invocation."""
    source = np.asarray(representation, dtype=np.float32)
    if source.shape != (MODEL_WINDOW_FRAMES, FEATURE_COUNT):
        raise RohmCorrectionError("RoHM window must have shape [144, 294]")
    visible = np.asarray(joint_visibility, dtype=bool)
    if visible.shape != (MODEL_WINDOW_FRAMES, 22):
        raise RohmCorrectionError("RoHM visibility must have shape [144, 22]")
    observed_root = (
        visible[:, 0]
        if root_observed is None
        else np.asarray(root_observed, dtype=bool)
    )
    if observed_root.shape != (MODEL_WINDOW_FRAMES,):
        raise RohmCorrectionError("RoHM root observations must have shape [144]")
    normalized = runtime.normalize(source)
    trajectory_condition = normalized[:, TRAJECTORY_ABSOLUTE_INDICES][None]
    trajectory_condition *= visible[:, 0][None, :, None]
    rng = np.random.RandomState(int(seed))
    trajectory_first = yield from _iter_runtime_pass(
        runtime,
        "iter_run_trajnet",
        "run_trajnet",
        (
            trajectory_condition,
            np.zeros((1, MODEL_WINDOW_FRAMES, POSE_COUNT), dtype=np.float32),
            rng,
        ),
        {"control_gate": 0.0},
        "RoHM: first trajectory pass",
        0,
        100,
    )
    pose_condition = _replace_absolute_trajectory(normalized, trajectory_first, runtime)
    pose_condition *= visibility_vector(visible)
    pose_condition[:, 290:294] = 0.0
    pose_first = yield from _iter_runtime_pass(
        runtime,
        "iter_run_posenet",
        "run_posenet",
        (pose_condition.T[None, :, None, :], rng),
        {},
        "RoHM: first pose pass",
        100,
        1000,
    )
    pose_first_time = pose_first[0, :, 0, :].T
    trajectory_second = yield from _iter_runtime_pass(
        runtime,
        "iter_run_trajnet",
        "run_trajnet",
        (
            trajectory_first,
            pose_first_time[:, TRAJECTORY_COUNT:][None],
            rng,
        ),
        {"control_gate": 1.0},
        "RoHM: controlled trajectory pass",
        1100,
        100,
    )
    pose_second_condition = _replace_absolute_trajectory(
        pose_first_time, trajectory_second, runtime
    )
    pose_second = yield from _iter_runtime_pass(
        runtime,
        "iter_run_posenet",
        "run_posenet",
        (pose_second_condition.T[None, :, None, :], rng),
        {},
        "RoHM: final pose pass",
        1200,
        1000,
    )
    corrected = runtime.denormalize(pose_second[0, :, 0, :].T)
    # Root confidence controls whether the noisy observation is masked from the
    # pose model; it must not disable TrajNet's correction of that observation.
    # Availability is tracked separately so a low-confidence root can still be
    # repaired while clips without any root observation remain in place.
    root_applied = bool(
        correct_root and root_motion_mode == "AUTOMATIC" and np.any(observed_root)
    )
    # Official RoHM applies PoseNet's complete reconstruction. Mixing source
    # rotations back per visible joint can break the globally coherent pose at
    # confidence boundaries, so the only supported pose policy is full output.
    # The independent pose toggle still lets TrajControl use reconstructed pose
    # to repair the root without applying those pose rotations to the target.
    if not correct_pose:
        corrected[:, TRAJECTORY_COUNT:] = source[:, TRAJECTORY_COUNT:]
    if not root_applied:
        corrected[:, :TRAJECTORY_COUNT] = source[:, :TRAJECTORY_COUNT]
    # Shape is deliberately neutral and never applied to the target rig.
    corrected[:, BETAS_SLICE] = 0.0
    warnings = _validate_corrected(source, corrected)
    return CorrectionResult(
        representation=corrected.astype(np.float32),
        joints=recover_joints(corrected),
        root_applied=root_applied,
        warnings=warnings,
    )


def correct_window(
    representation: np.ndarray,
    joint_visibility: np.ndarray,
    runtime,
    *,
    root_motion_mode: str,
    correct_pose: bool = True,
    correct_root: bool = True,
    seed: int = 10,
    root_observed: np.ndarray | None = None,
) -> CorrectionResult:
    """Synchronous one-window API retained for tests and scripts."""
    iterator = iter_correct_window(
        representation,
        joint_visibility,
        runtime,
        root_motion_mode=root_motion_mode,
        correct_pose=correct_pose,
        correct_root=correct_root,
        seed=seed,
        root_observed=root_observed,
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


def _channel_for_semantic(sample, semantic_id: str):
    return next(
        (
            channel
            for channel in sample.bones.values()
            if channel.semantic_id == semantic_id and channel.key_rotation
        ),
        None,
    )


def _root_channel(sample):
    return next(
        (
            channel
            for channel in sample.bones.values()
            if channel.semantic_id == "ROOT" and channel.key_location
        ),
        None,
    )


def _source_root(sample) -> Vector:
    if sample.grounding is not None:
        return sample.grounding.root.copy()
    if sample.root_object is not None:
        return sample.root_object[0].copy()
    channel = _root_channel(sample)
    if channel is not None:
        return channel.location.copy()
    if sample.grounding is not None:
        return sample.grounding.root.copy()
    return Vector()


def _effective_fps(timestamps: Sequence[float]) -> float:
    intervals = sorted(
        float(right) - float(left)
        for left, right in zip(timestamps, timestamps[1:], strict=False)
        if right > left
    )
    return 1.0 / intervals[len(intervals) // 2] if intervals else 0.0


def _uniform_timeline(timestamps: Sequence[float]) -> list[float]:
    """Return a 30 Hz clock with one look-ahead frame for representation velocity."""
    start, end = float(timestamps[0]), float(timestamps[-1])
    step = 1.0 / MODEL_FPS
    count = max(2, int(math.ceil(max(end - start, 0.0) / step)) + 2)
    return [start + index * step for index in range(count)]


def _sample_vector(
    values: Sequence[Vector], timestamps: Sequence[float], timestamp: float
) -> Vector:
    if timestamp <= timestamps[0]:
        return values[0].copy()
    if timestamp >= timestamps[-1]:
        return values[-1].copy()
    right = bisect.bisect_right(timestamps, timestamp)
    left = right - 1
    span = float(timestamps[right]) - float(timestamps[left])
    factor = 0.0 if span <= 1.0e-9 else (timestamp - timestamps[left]) / span
    return values[left].lerp(values[right], factor)


def _sample_float(
    values: Sequence[float], timestamps: Sequence[float], timestamp: float
) -> float:
    if timestamp <= timestamps[0]:
        return float(values[0])
    if timestamp >= timestamps[-1]:
        return float(values[-1])
    right = bisect.bisect_right(timestamps, timestamp)
    left = right - 1
    span = float(timestamps[right]) - float(timestamps[left])
    factor = 0.0 if span <= 1.0e-9 else (timestamp - timestamps[left]) / span
    return float(values[left]) + (float(values[right]) - float(values[left])) * factor


def _sample_quaternion(
    values: Sequence[Quaternion], timestamps: Sequence[float], timestamp: float
) -> Quaternion:
    if timestamp <= timestamps[0]:
        return values[0].copy()
    if timestamp >= timestamps[-1]:
        return values[-1].copy()
    right = bisect.bisect_right(timestamps, timestamp)
    left = right - 1
    span = float(timestamps[right]) - float(timestamps[left])
    factor = 0.0 if span <= 1.0e-9 else (timestamp - timestamps[left]) / span
    return values[left].slerp(values[right], factor).normalized()


def _quat_to_matrix(value: Quaternion) -> np.ndarray:
    return np.asarray([list(row) for row in value.to_matrix()], dtype=np.float32)


def _matrix_to_quaternion(value: np.ndarray) -> Quaternion:
    matrix = Matrix(tuple(tuple(float(item) for item in row) for row in value))
    result = matrix.to_quaternion()
    if result.magnitude < 1.0e-8:
        raise RohmCorrectionError("RoHM produced a degenerate rotation delta")
    return result.normalized()


def _blend_quaternions(values: Sequence[tuple[Quaternion, float]]) -> Quaternion:
    result = values[0][0].copy()
    total = max(float(values[0][1]), 1.0e-12)
    for value, raw_weight in values[1:]:
        candidate = value.copy()
        if result.dot(candidate) < 0.0:
            candidate.negate()
        weight = max(float(raw_weight), 0.0)
        result = result.slerp(candidate, weight / (total + weight)).normalized()
        total += weight
    return result


def _bounded_global_pose_proposal(
    source: Quaternion,
    proposal: Quaternion,
    maximum_degrees: float = MAX_LEARNED_GLOBAL_POSE_DELTA_DEGREES,
) -> Quaternion:
    """Keep a learned anatomical-global proposal close to deterministic repair."""
    baseline = source.normalized()
    candidate = proposal.normalized()
    angle = baseline.rotation_difference(candidate).angle
    maximum = math.radians(max(float(maximum_degrees), 0.0))
    if angle <= maximum or angle <= 1.0e-12:
        return candidate
    return baseline.slerp(candidate, maximum / angle).normalized()


def _window_weights(length: int, overlap: int = 30) -> np.ndarray:
    weights = np.ones(length, dtype=np.float64)
    edge = min(overlap, max(0, length // 2))
    for index in range(edge):
        value = 0.5 - 0.5 * math.cos(math.pi * (index + 1) / (edge + 1))
        weights[index] = min(weights[index], value)
        weights[-index - 1] = min(weights[-index - 1], value)
    return weights


def _window_starts(frame_count: int, overlap: int = 30) -> list[int]:
    last = max(0, frame_count - SOURCE_WINDOW_FRAMES)
    step = SOURCE_WINDOW_FRAMES - overlap
    starts = list(range(0, last + 1, step)) or [0]
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iter_correct_samples_impl(
    samples: Sequence[object],
    timestamps: Sequence[float],
    runtime,
    *,
    root_motion_mode: str,
    correct_pose: bool = True,
    correct_root: bool = True,
    visibility_threshold: float = 0.5,
    seed: int = 10,
) -> Iterator[tuple[str, int, int]]:
    """Correct a bake cooperatively using overlapping 145-frame windows."""
    if len(samples) != len(timestamps) or not samples:
        raise RohmCorrectionError("RoHM requires a non-empty timestamped sequence")
    source_times = [float(value) for value in timestamps]
    if any(
        not math.isfinite(value) or index and value <= source_times[index - 1]
        for index, value in enumerate(source_times)
    ):
        raise RohmCorrectionError("RoHM timestamps must be finite and increasing")
    threshold = min(max(float(visibility_threshold), 0.0), 1.0)
    missing = sorted(
        semantic
        for semantic in SMPLX_SEMANTICS
        if any(semantic not in sample.rohm_joint_offsets for sample in samples)
    )
    if missing:
        raise RohmCorrectionError(
            "RoHM needs complete body mappings; missing: " + ", ".join(missing)
        )

    model_times = _uniform_timeline(source_times)
    while len(model_times) < SOURCE_WINDOW_FRAMES:
        model_times.append(model_times[-1] + 1.0 / MODEL_FPS)
    source_roots = [_source_root(sample) for sample in samples]
    positions = np.zeros((len(model_times), 22, 3), dtype=np.float32)
    rotations = np.zeros((len(model_times), 21, 3, 3), dtype=np.float32)
    confidence = np.zeros((len(model_times), 22), dtype=np.float32)

    source_offsets = {
        semantic: [sample.rohm_joint_offsets[semantic] for sample in samples]
        for semantic in SMPLX_SEMANTICS
    }
    source_global_rotations: dict[str, list[Quaternion]] = {}
    source_confidence: dict[str, list[float]] = {}
    for semantic in SMPLX_SEMANTICS:
        semantic_rotations = []
        semantic_confidence = []
        previous = Quaternion()
        for sample in samples:
            value = sample.rohm_rotations.get(semantic, previous).copy()
            if value.magnitude < 1.0e-8:
                value = Quaternion()
            value.normalize()
            if semantic_rotations and semantic_rotations[-1].dot(value) < 0.0:
                value.negate()
            previous = value.copy()
            semantic_rotations.append(value)
            semantic_confidence.append(
                min(max(float(sample.rohm_confidence.get(semantic, 0.0)), 0.0), 1.0)
            )
        source_global_rotations[semantic] = semantic_rotations
        source_confidence[semantic] = semantic_confidence

    root_confidence = [
        min(max(float(sample.rohm_root_confidence), 0.0), 1.0) for sample in samples
    ]
    global_rotations = np.zeros((len(model_times), 22, 3, 3), dtype=np.float32)
    for frame_index, timestamp in enumerate(model_times):
        root = _sample_vector(source_roots, source_times, timestamp)
        for joint, semantic in enumerate(SMPLX_SEMANTICS):
            offset = _sample_vector(source_offsets[semantic], source_times, timestamp)
            positions[frame_index, joint] = tuple(root + offset)
            confidence[frame_index, joint] = _sample_float(
                source_confidence[semantic], source_times, timestamp
            )
            global_rotations[frame_index, joint] = _quat_to_matrix(
                _sample_quaternion(
                    source_global_rotations[semantic], source_times, timestamp
                )
            )
        confidence[frame_index, 0] = _sample_float(
            root_confidence, source_times, timestamp
        )
    rotations[:] = global_anatomical_to_local(global_rotations)

    body_height = max(
        (float(getattr(sample, "body_height", 0.0) or 0.0) for sample in samples),
        default=0.0,
    )
    if body_height <= 1.0e-4:
        raise RohmCorrectionError("RoHM target-rig body height is unavailable")
    candidates: list[list[tuple[list[Quaternion], np.ndarray, float]]] = [
        [] for _ in model_times
    ]
    corrected_model = np.zeros(len(model_times), dtype=np.bool_)
    quality_warnings: list[str] = []
    starts = _window_starts(len(model_times))
    inference_units = len(starts) * _WINDOW_INFERENCE_UNITS
    total_units = inference_units + len(samples)
    yield "RoHM: preparing 30 FPS semantic motion", 0, total_units
    for window_index, start in enumerate(starts):
        stop = start + SOURCE_WINDOW_FRAMES
        source_repr, context = build_representation(
            positions[start:stop],
            rotations[start:stop],
            body_height=body_height,
        )
        visible = confidence[start : start + MODEL_WINDOW_FRAMES] >= threshold
        window_iterator = iter_correct_window(
            source_repr,
            visible,
            runtime,
            root_motion_mode=root_motion_mode,
            correct_pose=correct_pose,
            correct_root=correct_root,
            seed=seed + window_index,
            root_observed=confidence[start : start + MODEL_WINDOW_FRAMES, 0] > 0.0,
        )
        while True:
            try:
                phase, current, _total = next(window_iterator)
            except StopIteration as stopped:
                result = stopped.value
                break
            yield (
                f"{phase} · window {window_index + 1}/{len(starts)}",
                window_index * _WINDOW_INFERENCE_UNITS + int(current),
                total_units,
            )
        for warning in result.warnings:
            if warning not in quality_warnings:
                quality_warnings.append(warning)
        source_local = rotation_6d_to_matrix(
            source_repr[:, 154:280].reshape(-1, 6)
        ).reshape(MODEL_WINDOW_FRAMES, 21, 3, 3)
        corrected_local = rotation_6d_to_matrix(
            result.representation[:, 154:280].reshape(-1, 6)
        ).reshape(MODEL_WINDOW_FRAMES, 21, 3, 3)
        source_joints = recover_joints(source_repr, context)
        corrected_joints = recover_joints(result.representation, context)
        root_delta = corrected_joints[:, 0] - source_joints[:, 0]
        weights = _window_weights(MODEL_WINDOW_FRAMES)
        for local_index in range(MODEL_WINDOW_FRAMES):
            global_index = start + local_index
            output_locals = [
                _matrix_to_quaternion(corrected_local[local_index, joint])
                for joint in range(21)
            ]
            source_locals = [
                _matrix_to_quaternion(source_local[local_index, joint])
                for joint in range(21)
            ]
            changed = bool(
                np.linalg.norm(root_delta[local_index]) > 1.0e-6
                or any(
                    source.rotation_difference(output).angle > 1.0e-5
                    for source, output in zip(source_locals, output_locals, strict=True)
                )
            )
            corrected_model[global_index] |= changed
            candidates[global_index].append(
                (output_locals, root_delta[local_index], float(weights[local_index]))
            )

    model_output_locals: list[list[Quaternion]] = [
        [Quaternion() for _ in model_times] for _ in range(21)
    ]
    model_root_delta = [Vector() for _ in model_times]
    for frame_index, frame_candidates in enumerate(candidates):
        if not frame_candidates:
            continue
        for joint in range(21):
            model_output_locals[joint][frame_index] = _blend_quaternions(
                [(value[0][joint], value[2]) for value in frame_candidates]
            )
        total_weight = max(sum(value[2] for value in frame_candidates), 1.0e-12)
        root = (
            sum(
                (value[1] * value[2] for value in frame_candidates),
                np.zeros(3, dtype=np.float32),
            )
            / total_weight
        )
        model_root_delta[frame_index] = Vector(tuple(float(item) for item in root))

    corrected_source_frames = 0
    source_root_deltas: list[tuple[float, float, float]] = []
    source_pose_global_rotations: list[dict[str, Quaternion]] = []
    prediction_times = model_times[: len(model_output_locals[0])]
    model_source_locals = [
        [_matrix_to_quaternion(value) for value in rotations[:, joint]]
        for joint in range(21)
    ]
    for source_index, (sample, timestamp) in enumerate(
        zip(samples, source_times, strict=True), 1
    ):
        applied = False
        selected_locals = np.zeros((1, 21, 3, 3), dtype=np.float32)
        untrusted_semantics: list[str] = []
        for joint, semantic in enumerate(SMPLX_SEMANTICS[1:]):
            source_local = _sample_quaternion(
                model_source_locals[joint],
                prediction_times,
                timestamp,
            )
            output_local = _sample_quaternion(
                model_output_locals[joint], prediction_times, timestamp
            )
            trusted = float(sample.rohm_confidence.get(semantic, 0.0)) >= threshold
            selected = source_local if trusted or not correct_pose else output_local
            selected_locals[0, joint] = _quat_to_matrix(selected)
            if not trusted and correct_pose:
                untrusted_semantics.append(semantic)
        if untrusted_semantics:
            proposed_globals = local_to_global_anatomical(selected_locals)[0]
            guarded_globals: dict[str, Quaternion] = {}
            for semantic in untrusted_semantics:
                joint = SMPLX_SEMANTICS.index(semantic)
                source_global = _sample_quaternion(
                    source_global_rotations[semantic], source_times, timestamp
                )
                proposal_global = _matrix_to_quaternion(proposed_globals[joint])
                guarded = _bounded_global_pose_proposal(source_global, proposal_global)
                if source_global.rotation_difference(guarded).angle > 1.0e-5:
                    guarded_globals[semantic] = guarded
            applied = bool(guarded_globals)
            source_pose_global_rotations.append(guarded_globals)
        else:
            source_pose_global_rotations.append({})
        if correct_root and root_motion_mode == "AUTOMATIC":
            delta = _sample_vector(model_root_delta, prediction_times, timestamp)
            if delta.length > 1.0e-6:
                applied |= bool(
                    sample.root_object is not None or _root_channel(sample) is not None
                )
            source_root_deltas.append(tuple(float(value) for value in delta))
        else:
            source_root_deltas.append((0.0, 0.0, 0.0))
        corrected_source_frames += int(applied)
        yield (
            "RoHM: applying corrected motion",
            inference_units + source_index,
            total_units,
        )

    source_fps = _effective_fps(source_times)
    observed = sum(
        float(sample.rohm_confidence.get(semantic, 0.0)) >= threshold
        for sample in samples
        for semantic in SMPLX_SEMANTICS
    )
    observed_fraction = observed / max(len(samples) * len(SMPLX_SEMANTICS), 1)
    return CorrectionStats(
        source_fps=source_fps,
        model_frames=len(model_times),
        windows=len(starts),
        corrected_source_frames=corrected_source_frames,
        corrected_model_frames=int(corrected_model.sum()),
        low_rate_warning=bool(source_fps and source_fps < MODEL_FPS - 1.0e-3),
        root_deltas=tuple(source_root_deltas),
        observed_fraction=float(observed_fraction),
        warnings=tuple(quality_warnings),
        pose_global_rotations=tuple(source_pose_global_rotations),
    )


def correct_samples(
    samples: Sequence[object],
    timestamps: Sequence[float],
    runtime,
    *,
    root_motion_mode: str,
    correct_pose: bool = True,
    correct_root: bool = True,
    visibility_threshold: float = 0.5,
    seed: int = 10,
) -> CorrectionStats:
    """Synchronous timestamped-bake API retained for scripts and tests."""
    iterator = _iter_correct_samples_impl(
        samples,
        timestamps,
        runtime,
        root_motion_mode=root_motion_mode,
        correct_pose=correct_pose,
        correct_root=correct_root,
        visibility_threshold=visibility_threshold,
        seed=seed,
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


def iter_correct_samples(*args, **kwargs) -> Iterator[object]:
    """Cooperative baked-sequence entry point used by Blender."""
    return (yield from _iter_correct_samples_impl(*args, **kwargs))


__all__ = [
    "CorrectionResult",
    "CorrectionStats",
    "RepresentationContext",
    "RohmCorrectionError",
    "SMPLX_PARENTS",
    "SMPLX_SEMANTICS",
    "build_representation",
    "correct_samples",
    "correct_window",
    "global_anatomical_to_local",
    "iter_correct_samples",
    "iter_correct_window",
    "matrix_to_rotation_6d",
    "local_to_global_anatomical",
    "recover_joints",
    "rotation_6d_to_matrix",
]
