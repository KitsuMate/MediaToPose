"""The SOMA 30-joint skeleton that NVIDIA Kimodo generates motion on.

Joint names, hierarchy and the neutral (T-pose) joint positions come from
``kimodo/skeleton/definitions.py`` and ``kimodo/assets/skeletons/somaskel30``
in https://github.com/nv-tlabs/kimodo at revision
1aece8c124d73d255ceff5086d983b844c9f4e94 (Apache-2.0, Copyright (c) 2026
NVIDIA CORPORATION & AFFILIATES). The positions were read once from the
upstream tensor storage and rounded to micrometres.

Kimodo space is Y up, +Z forward and metres; the character's left is +X.
The anatomical frame used by the retargeting code is (right, forward, up),
Z up, so ``KIMODO_TO_ANATOMICAL`` maps Kimodo ``(x, y, z)`` to ``(-x, z, y)``.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumbEnd",
    "LeftHandMiddleEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumbEnd",
    "RightHandMiddleEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)
JOINT_COUNT = len(JOINT_NAMES)
JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(JOINT_NAMES)}

_PARENT_NAMES: dict[str, str | None] = {
    "Hips": None,
    "Spine1": "Hips",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    "Neck1": "Chest",
    "Neck2": "Neck1",
    "Head": "Neck2",
    "Jaw": "Head",
    "LeftEye": "Head",
    "RightEye": "Head",
    "LeftShoulder": "Chest",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "LeftHandThumbEnd": "LeftHand",
    "LeftHandMiddleEnd": "LeftHand",
    "RightShoulder": "Chest",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "RightHandThumbEnd": "RightHand",
    "RightHandMiddleEnd": "RightHand",
    "LeftLeg": "Hips",
    "LeftShin": "LeftLeg",
    "LeftFoot": "LeftShin",
    "LeftToeBase": "LeftFoot",
    "RightLeg": "Hips",
    "RightShin": "RightLeg",
    "RightFoot": "RightShin",
    "RightToeBase": "RightFoot",
}
# Parent index per joint; the root (Hips) has -1. Parents precede children.
PARENTS: tuple[int, ...] = tuple(
    -1 if (parent := _PARENT_NAMES[name]) is None else JOINT_INDEX[parent]
    for name in JOINT_NAMES
)
ROOT = JOINT_INDEX["Hips"]

# Neutral T-pose joint positions relative to Hips, Kimodo space, metres.
REST_POSITIONS = np.array(
    (
        (0.0, 0.0, 0.0),
        (-0.000137, 0.050038, -0.000537),
        (-0.000137, 0.121291, -0.000836),
        (-0.000137, 0.196791, -0.008995),
        (-0.001954, 0.459904, -0.014529),
        (-0.001954, 0.536998, 0.008497),
        (-0.001954, 0.598287, 0.028034),
        (-0.001928, 0.603043, 0.058984),
        (0.03011, 0.652089, 0.103903),
        (-0.034179, 0.651906, 0.103617),
        (0.016079, 0.429163, 0.042139),
        (0.165278, 0.429163, -0.012884),
        (0.452671, 0.429163, -0.01291),
        (0.723611, 0.429163, -0.012884),
        (0.846297, 0.396961, 0.035447),
        (0.91373, 0.426034, -0.013224),
        (-0.013938, 0.428594, 0.043146),
        (-0.16431, 0.428594, -0.01231),
        (-0.451677, 0.428594, -0.012336),
        (-0.723013, 0.428594, -0.01231),
        (-0.845655, 0.39648, 0.035731),
        (-0.913019, 0.425528, -0.012625),
        (0.100432, -0.084345, 0.025957),
        (0.100432, -0.516563, 0.017927),
        (0.100432, -0.938114, -0.016888),
        (0.100432, -0.988708, 0.115427),
        (-0.100473, -0.082953, 0.026203),
        (-0.100473, -0.516575, 0.018148),
        (-0.100473, -0.937749, -0.016636),
        (-0.100473, -0.988545, 0.116206),
    ),
    dtype=np.float64,
)
REST_POSITIONS.setflags(write=False)
# Hips height above the lowest toe in the neutral pose.
REST_HIP_HEIGHT = float(-REST_POSITIONS[:, 1].min())

# Heading is measured from the hip joints, ordered (right, left).
HIP_JOINTS = (JOINT_INDEX["RightLeg"], JOINT_INDEX["LeftLeg"])
# End-effector chains in chain order; the first joint carries the rotation.
EFFECTOR_CHAINS: dict[str, tuple[int, ...]] = {
    "LEFT_HAND": (JOINT_INDEX["LeftHand"], JOINT_INDEX["LeftHandMiddleEnd"]),
    "RIGHT_HAND": (JOINT_INDEX["RightHand"], JOINT_INDEX["RightHandMiddleEnd"]),
    "LEFT_FOOT": (JOINT_INDEX["LeftFoot"], JOINT_INDEX["LeftToeBase"]),
    "RIGHT_FOOT": (JOINT_INDEX["RightFoot"], JOINT_INDEX["RightToeBase"]),
    "HIPS": (ROOT,),
}
# Foot-contact channels in feature order: left heel, left toe, right heel,
# right toe.
CONTACT_JOINTS = (
    JOINT_INDEX["LeftFoot"],
    JOINT_INDEX["LeftToeBase"],
    JOINT_INDEX["RightFoot"],
    JOINT_INDEX["RightToeBase"],
)

KIMODO_TO_ANATOMICAL = np.array(
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float64
)
KIMODO_TO_ANATOMICAL.setflags(write=False)


def _depth_levels() -> tuple[np.ndarray, ...]:
    depth = [0] * JOINT_COUNT
    for index, parent in enumerate(PARENTS):
        if parent >= 0:
            depth[index] = depth[parent] + 1
    return tuple(
        np.asarray([i for i in range(JOINT_COUNT) if depth[i] == level], dtype=np.int64)
        for level in range(1, max(depth) + 1)
    )


_LEVELS = _depth_levels()
_PARENT_ARRAY = np.asarray(PARENTS, dtype=np.int64)
_BONE_OFFSETS = REST_POSITIONS - np.where(
    _PARENT_ARRAY[:, None] >= 0, REST_POSITIONS[np.maximum(_PARENT_ARRAY, 0)], 0.0
)


def forward_kinematics(
    local_rotations: np.ndarray, root_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Global rotations [..., 30, 3, 3] and positions [..., 30, 3].

    ``local_rotations`` are parent-relative rotation matrices [..., 30, 3, 3]
    (the root's is global) and ``root_positions`` the Hips positions [..., 3].
    Identity rotations give the neutral T-pose.
    """
    local = np.asarray(local_rotations, dtype=np.float64)
    root = np.asarray(root_positions, dtype=np.float64)
    global_rotations = np.empty_like(local)
    positions = np.empty(local.shape[:-1], dtype=np.float64)
    global_rotations[..., ROOT, :, :] = local[..., ROOT, :, :]
    positions[..., ROOT, :] = root
    for joints in _LEVELS:
        parents = _PARENT_ARRAY[joints]
        parent_rotations = global_rotations[..., parents, :, :]
        global_rotations[..., joints, :, :] = (
            parent_rotations @ local[..., joints, :, :]
        )
        positions[..., joints, :] = positions[..., parents, :] + np.einsum(
            "...jab,jb->...ja", parent_rotations, _BONE_OFFSETS[joints]
        )
    return global_rotations, positions


def global_to_local(global_rotations: np.ndarray) -> np.ndarray:
    """Parent-relative rotations from global rotations [..., 30, 3, 3]."""
    rotations = np.asarray(global_rotations, dtype=np.float64)
    local = rotations.copy()
    children = np.arange(JOINT_COUNT) != ROOT
    parents = _PARENT_ARRAY[children]
    local[..., children, :, :] = (
        np.swapaxes(rotations[..., parents, :, :], -1, -2)
        @ rotations[..., children, :, :]
    )
    return local


def heading_angle(positions: np.ndarray) -> np.ndarray:
    """Facing angle about +Y from the hip joints of positions [..., 30, 3].

    Zero faces +Z; the facing direction is ``(sin a, 0, cos a)``.
    """
    joints = np.asarray(positions, dtype=np.float64)
    right, left = HIP_JOINTS
    across = joints[..., right, :] - joints[..., left, :]
    return np.arctan2(across[..., 2], -across[..., 0])


def steady_heading(positions: np.ndarray, smoothing: float = 4.0) -> np.ndarray:
    """Facing angles [T] from the hip joints, unwrapped and filled in where
    the hip line is nearly vertical (lying on a side, mid-roll), so the
    heading never jumps, then low-passed (Gaussian, ``smoothing`` frames) so
    it does not sway with the pelvis on every stride.

    Frames whose hip line keeps less than half its length on the ground
    take the angle interpolated between their reliable neighbours.
    """
    joints = np.asarray(positions, dtype=np.float64)
    right, left = HIP_JOINTS
    across = joints[:, right, :] - joints[:, left, :]
    ground = np.hypot(across[:, 0], across[:, 2])
    reliable = ground > 0.5 * np.linalg.norm(across, axis=-1)
    raw = np.arctan2(across[:, 2], -across[:, 0])
    frames = np.arange(len(raw))
    if not reliable.any():
        return np.zeros(len(raw))
    angles = np.interp(frames, frames[reliable], np.unwrap(raw[reliable]))
    if smoothing <= 0.0 or len(angles) < 3:
        return angles
    radius = int(np.ceil(3.0 * smoothing))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / smoothing) ** 2)
    padded = np.pad(angles, radius, mode="edge")
    return np.convolve(padded, kernel / kernel.sum(), mode="valid")


def rotation_y(angle: np.ndarray | float) -> np.ndarray:
    """Rotation matrices [..., 3, 3] about +Y."""
    angles = np.asarray(angle, dtype=np.float64)
    cos, sin = np.cos(angles), np.sin(angles)
    zero, one = np.zeros_like(angles), np.ones_like(angles)
    return np.stack(
        (cos, zero, sin, zero, one, zero, -sin, zero, cos), axis=-1
    ).reshape((*angles.shape, 3, 3))


__all__ = [
    "CONTACT_JOINTS",
    "EFFECTOR_CHAINS",
    "HIP_JOINTS",
    "JOINT_COUNT",
    "JOINT_INDEX",
    "JOINT_NAMES",
    "KIMODO_TO_ANATOMICAL",
    "PARENTS",
    "REST_HIP_HEIGHT",
    "REST_POSITIONS",
    "ROOT",
    "forward_kinematics",
    "global_to_local",
    "heading_angle",
    "rotation_y",
    "steady_heading",
]
