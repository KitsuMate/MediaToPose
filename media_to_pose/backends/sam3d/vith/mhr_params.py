"""Decoder outputs to MHR body and hand parameters."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..mhr_runtime import MhrModel
from .common import VithRuntimeError
from .rotations import euler_xyz_to_matrix, matrix_to_euler_zyx, rotation_6d_to_matrix

BODY_3DOF_INDICES = np.asarray(
    (
        (0, 2, 4),
        (6, 8, 10),
        (12, 13, 14),
        (15, 16, 17),
        (18, 19, 20),
        (21, 22, 23),
        (24, 25, 26),
        (27, 28, 29),
        (34, 35, 36),
        (37, 38, 39),
        (44, 45, 46),
        (53, 54, 55),
        (64, 65, 66),
        (85, 69, 73),
        (86, 70, 79),
        (87, 71, 82),
        (88, 72, 76),
        (91, 92, 93),
        (112, 96, 100),
        (113, 97, 106),
        (114, 98, 109),
        (115, 99, 103),
        (130, 131, 132),
    ),
    dtype=np.intp,
)


BODY_1DOF_INDICES = np.asarray(
    (
        1,
        3,
        5,
        7,
        9,
        11,
        30,
        31,
        32,
        33,
        40,
        41,
        42,
        43,
        47,
        48,
        49,
        50,
        51,
        52,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        67,
        68,
        74,
        75,
        77,
        78,
        80,
        81,
        83,
        84,
        89,
        90,
        94,
        95,
        101,
        102,
        104,
        105,
        107,
        108,
        110,
        111,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
    ),
    dtype=np.intp,
)


BODY_TRANSLATION_INDICES = np.arange(124, 130, dtype=np.intp)
BODY_HAND_INDICES = np.arange(62, 116, dtype=np.intp)
HAND_DOFS = (3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1)
HAND_MODEL_3DOF_MASK = np.concatenate(
    [np.full(value, value == 3, dtype=bool) for value in HAND_DOFS]
)


HAND_MODEL_1DOF_MASK = ~HAND_MODEL_3DOF_MASK
HAND_CONT_3DOF_MASK = np.concatenate(
    [np.full(2 * value, value == 3, dtype=bool) for value in HAND_DOFS]
)


HAND_CONT_1DOF_MASK = ~HAND_CONT_3DOF_MASK


def compact_cont_to_body(value: object) -> np.ndarray:
    continuous = np.asarray(value, dtype=np.float32)
    if continuous.shape[-1:] != (260,):
        raise VithRuntimeError(
            f"Body continuous pose must be [...,260], got {continuous.shape}"
        )
    count_3d = BODY_3DOF_INDICES.size
    count_1d = len(BODY_1DOF_INDICES)
    rotations = continuous[..., : 2 * count_3d].reshape(*continuous.shape[:-1], -1, 6)
    eulers = matrix_to_euler_zyx(rotation_6d_to_matrix(rotations))[..., ::-1]
    one_d = continuous[..., 2 * count_3d : 2 * count_3d + 2 * count_1d]
    one_d = one_d.reshape(*continuous.shape[:-1], -1, 2)
    one_d = np.arctan2(one_d[..., 0], one_d[..., 1])
    translations = continuous[..., 2 * count_3d + 2 * count_1d :]
    result = np.zeros((*continuous.shape[:-1], 133), dtype=np.float32)
    result[..., BODY_3DOF_INDICES.reshape(-1)] = eulers.reshape(
        *continuous.shape[:-1], -1
    )
    result[..., BODY_1DOF_INDICES] = one_d
    result[..., BODY_TRANSLATION_INDICES] = translations
    result[..., BODY_HAND_INDICES] = 0.0
    result[..., -3:] = 0.0
    return result


def compact_cont_to_hand(value: object) -> np.ndarray:
    continuous = np.asarray(value, dtype=np.float32)
    if continuous.shape[-1:] != (54,):
        raise VithRuntimeError(
            f"Hand continuous pose must be [...,54], got {continuous.shape}"
        )
    result = np.zeros((*continuous.shape[:-1], 27), dtype=np.float32)
    three = continuous[..., HAND_CONT_3DOF_MASK].reshape(*continuous.shape[:-1], -1, 6)
    result[..., HAND_MODEL_3DOF_MASK] = matrix_to_euler_zyx(
        rotation_6d_to_matrix(three)
    )[..., ::-1].reshape(*continuous.shape[:-1], -1)
    one = continuous[..., HAND_CONT_1DOF_MASK].reshape(*continuous.shape[:-1], -1, 2)
    result[..., HAND_MODEL_1DOF_MASK] = np.arctan2(one[..., 0], one[..., 1])
    return result


def _mhr_model_parameters(
    model: MhrModel,
    *,
    global_rotation: np.ndarray,
    body_pose: np.ndarray,
    hand_pose: np.ndarray,
    scale: np.ndarray,
    hand_model: bool,
) -> tuple[np.ndarray, np.ndarray]:
    batch = len(global_rotation)
    global_translation = np.zeros((batch, 3), dtype=np.float32)
    effective_rotation = global_rotation
    if hand_model:
        source_matrix = euler_xyz_to_matrix(global_rotation)
        effective_matrix = source_matrix @ model.local_to_world_wrist
        effective_rotation = matrix_to_euler_zyx(effective_matrix)[..., ::-1]
        wrist_offset = model.right_wrist_coords - model.root_coords
        global_translation = -(effective_matrix @ wrist_offset + model.root_coords)
    full_pose = np.concatenate(
        (global_translation * np.float32(10.0), effective_rotation, body_pose[:, :130]),
        axis=1,
    ).astype(np.float32)
    left, right = np.split(hand_pose, 2, axis=1)
    left_parameters = compact_cont_to_hand(
        model.hand_pose_mean + left @ model.hand_pose_components
    )
    right_parameters = compact_cont_to_hand(
        model.hand_pose_mean + right @ model.hand_pose_components
    )
    full_pose[:, model.hand_joint_indices_left] = left_parameters
    full_pose[:, model.hand_joint_indices_right] = right_parameters
    actual_scale = model.scale_mean + scale @ model.scale_components
    parameters = np.concatenate((full_pose, actual_scale), axis=1).astype(np.float32)
    if hand_model:
        parameters[:, model.nonhand_param_indices] = 0.0
    return parameters, effective_rotation.astype(np.float32)


def decode_raw_mhr(
    model: MhrModel,
    raw_mhr_parameters: object,
    raw_camera: object,
    *,
    hand_model: bool = False,
) -> dict[str, np.ndarray]:
    """Decode the raw 519+3 learned head boundary and run native MHR."""

    raw = np.ascontiguousarray(raw_mhr_parameters, dtype=np.float32)
    camera = np.ascontiguousarray(raw_camera, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 519:
        raise VithRuntimeError(f"raw MHR head output must be [B,519], got {raw.shape}")
    if camera.shape != (raw.shape[0], 3):
        raise VithRuntimeError(f"raw camera output must be [B,3], got {camera.shape}")
    global_matrix = rotation_6d_to_matrix(raw[:, :6])
    global_rotation = matrix_to_euler_zyx(global_matrix)
    body_continuous = raw[:, 6:266]
    body_pose = compact_cont_to_body(body_continuous)
    shape = raw[:, 266:311].copy()
    scale = raw[:, 311:339].copy()
    hand = raw[:, 339:447].copy()
    face = np.zeros((len(raw), 72), dtype=np.float32)
    model_parameters, _ = _mhr_model_parameters(
        model,
        global_rotation=global_rotation,
        body_pose=body_pose,
        hand_pose=hand,
        scale=scale,
        hand_model=hand_model,
    )
    geometry = model.forward(shape, model_parameters, face).camera_space()
    keypoints = geometry.keypoints.copy()
    if hand_model:
        # The hand head is trained against the right-hand 21-point slice.
        keypoints[:, :21] = 0.0
        keypoints[:, 42:] = 0.0
    return {
        "raw_mhr_parameters": raw,
        "pred_pose_raw": raw[:, :266].copy(),
        "pred_cam": camera,
        "global_rot": global_rotation.astype(np.float32),
        "body_pose": body_pose,
        "shape": shape,
        "scale": scale,
        "hand": hand,
        "face": face,
        "pred_keypoints_3d": keypoints,
        "pred_vertices": geometry.vertices,
        "pred_joint_coords": geometry.joint_coords,
        "joint_global_rots": geometry.joint_rotations,
        "mhr_model_params": model_parameters,
    }


def previous_estimate(output: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate(
            (
                output["pred_pose_raw"],
                output["shape"],
                output["scale"],
                output["hand"],
                output["face"],
                output["pred_cam"],
            ),
            axis=1,
        ),
        dtype=np.float32,
    )


def _fused_local_wrist(
    predicted_global: np.ndarray, wrist_zero: np.ndarray
) -> np.ndarray:
    # Exact equivalent of upstream ``einsum('kabc,kabd->kadc', ...)``.
    return np.swapaxes(wrist_zero, -1, -2) @ predicted_global
