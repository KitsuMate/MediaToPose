"""Production NumPy orchestrator for the split SAM 3D Body ViT-H bundle.

The learned model is kept in 22 small, independently verifiable ONNX graphs
(one backbone plus 21 decoder components).  Crop geometry, camera projection,
MHR decoding, and body/hand fusion are deterministic NumPy operations.  This
module intentionally has no PyTorch, PyMomentum, Roma, or exporter dependency.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from . import sam3d_ort_runtime as _ort_runtime
from .core.log import get_logger
from .sam3d_mhr_runtime import MhrModel

LOGGER = get_logger(__name__)

IMAGE_SIZE = 512
BACKBONE_WIDTH = 384
BACKBONE_TRIM = 64
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)

OUTPUT_KEYS = (
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_cam_t",
    "focal_length",
    "global_rot",
    "scale_params",
    "pred_joint_coords",
    "pred_global_rots",
)

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


class VithRuntimeError(_ort_runtime.OrtRuntimeError):
    """The split bundle or a deterministic runtime input is incompatible."""


class GraphExecutionError(VithRuntimeError):
    """An ORT component session failed while executing a declared graph."""


@dataclass(slots=True)
class CropBatch:
    images: np.ndarray
    bbox_center: np.ndarray
    bbox_scale: np.ndarray
    affine: np.ndarray
    original_size: np.ndarray
    camera_intrinsics: np.ndarray


def _normalize(value: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, np.float32(epsilon))


def rotation_6d_to_matrix(value: object) -> np.ndarray:
    """Match the upstream first-two-columns Gram-Schmidt conversion."""

    data = np.asarray(value, dtype=np.float32)
    if data.shape[-1:] != (6,):
        raise VithRuntimeError(f"6-D rotations must end in 6, got {data.shape}")
    first = _normalize(data[..., :3])
    third = _normalize(np.cross(first, data[..., 3:]))
    second = np.cross(third, first)
    return np.stack((first, second, third), axis=-1).astype(np.float32, copy=False)


def matrix_to_euler_zyx(matrix: object) -> np.ndarray:
    """Intrinsic ZYX angles ``[z, y, x]`` used for the global MHR rotation."""

    value = np.asarray(matrix, dtype=np.float32)
    sy = np.sqrt(value[..., 0, 0] ** 2 + value[..., 1, 0] ** 2)
    singular = sy < np.float32(1e-6)
    x = np.arctan2(value[..., 2, 1], value[..., 2, 2])
    y = np.arctan2(-value[..., 2, 0], sy)
    z = np.arctan2(value[..., 1, 0], value[..., 0, 0])
    x_singular = np.arctan2(-value[..., 1, 2], value[..., 1, 1])
    x = np.where(singular, x_singular, x)
    z = np.where(singular, np.float32(0.0), z)
    return np.stack((z, y, x), axis=-1).astype(np.float32, copy=False)


def euler_xyz_to_matrix(angles: object) -> np.ndarray:
    """Extrinsic xyz / equivalent intrinsic ZYX rotation matrix."""

    value = np.asarray(angles, dtype=np.float32)
    x, y, z = np.moveaxis(value, -1, 0)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return (
        np.stack(
            (
                cy * cz,
                -cx * sz + sx * sy * cz,
                sx * sz + cx * sy * cz,
                cy * sz,
                cx * cz + sx * sy * sz,
                -sx * cz + cx * sy * sz,
                -sy,
                sx * cy,
                cx * cy,
            ),
            axis=-1,
        )
        .reshape(*value.shape[:-1], 3, 3)
        .astype(np.float32, copy=False)
    )


def euler_xzy_to_matrix(angles: object) -> np.ndarray:
    """Intrinsic XZY matrix for the two wrist joints."""

    value = np.asarray(angles, dtype=np.float32)
    x, z, y = np.moveaxis(value, -1, 0)
    cx, cz, cy = np.cos(x), np.cos(z), np.cos(y)
    sx, sz, sy = np.sin(x), np.sin(z), np.sin(y)
    return (
        np.stack(
            (
                cz * cy,
                -sz,
                cz * sy,
                cx * sz * cy + sx * sy,
                cx * cz,
                cx * sz * sy - sx * cy,
                sx * sz * cy - cx * sy,
                sx * cz,
                sx * sz * sy + cx * cy,
            ),
            axis=-1,
        )
        .reshape(*value.shape[:-1], 3, 3)
        .astype(np.float32, copy=False)
    )


def matrix_to_euler_xzy(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    z = np.arcsin(np.clip(-value[..., 0, 1], -1.0, 1.0))
    cos_z = np.cos(z)
    regular = np.abs(cos_z) > np.float32(1e-7)
    x = np.arctan2(value[..., 2, 1], value[..., 1, 1])
    y = np.arctan2(value[..., 0, 2], value[..., 0, 0])
    # Roma fixes the third angle to zero at gimbal lock.  Solve the remaining
    # first angle from the stable lower-right block.
    x_singular = np.arctan2(-value[..., 1, 2], value[..., 2, 2])
    return np.stack(
        (np.where(regular, x, x_singular), z, np.where(regular, y, 0.0)), axis=-1
    ).astype(np.float32, copy=False)


def rotation_angle_difference(first: object, second: object) -> np.ndarray:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    relative = a @ np.swapaxes(b, -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def fix_wrist_euler(value: object) -> np.ndarray:
    wrist = np.asarray(value, dtype=np.float32)
    x, z, y = np.moveaxis(wrist, -1, 0)
    alternate = np.stack(
        (
            np.arctan2(np.sin(x + np.pi), np.cos(x + np.pi)),
            np.arctan2(np.sin(-(z + np.pi)), np.cos(-(z + np.pi))),
            np.arctan2(np.sin(y + np.pi), np.cos(y + np.pi)),
        ),
        axis=-1,
    )
    limits = ((-2.2, 1.0), (-2.2, 1.5), (-1.2, 1.5))

    def violation(candidate):
        result = np.zeros(candidate.shape[:-1], dtype=np.float32)
        for index, (low, high) in enumerate(limits):
            item = candidate[..., index]
            result += np.maximum(low - item, 0.0) ** 2
            result += np.maximum(item - high, 0.0) ** 2
        return result

    use_alternate = violation(alternate) < violation(wrist)
    return np.where(use_alternate[..., None], alternate, wrist).astype(np.float32)


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


def _bbox_center_scale(
    boxes: np.ndarray, padding: float
) -> tuple[np.ndarray, np.ndarray]:
    center = (boxes[:, :2] + boxes[:, 2:]) * np.float32(0.5)
    raw_scale = (boxes[:, 2:] - boxes[:, :2]) * np.float32(padding)
    if np.any(raw_scale <= 0.0):
        raise VithRuntimeError("Bounding boxes must have positive width and height")
    # TopdownAffine first expands to 0.75 width/height, then to the square
    # 512x512 model input.  Its final side is max(h, w / 0.75).
    side = np.maximum(raw_scale[:, 1], raw_scale[:, 0] / np.float32(0.75))
    return center.astype(np.float32), np.repeat(side[:, None], 2, axis=1)


def prepare_crop_batch(
    image_rgb: object,
    bboxes_xyxy: object,
    *,
    padding: float = 1.25,
    camera_intrinsics: object | None = None,
) -> CropBatch:
    image = np.asarray(image_rgb)
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise VithRuntimeError(f"image_rgb must be HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        if not np.isfinite(image).all():
            raise VithRuntimeError("image_rgb contains NaN or infinity")
        image = np.clip(image, 0, 255).astype(np.uint8)
    if not len(boxes):
        raise VithRuntimeError("At least one person bounding box is required")
    if not np.isfinite(boxes).all():
        raise VithRuntimeError("Bounding boxes contain NaN or infinity")
    center, scale = _bbox_center_scale(boxes, padding)
    factor = np.float32(IMAGE_SIZE) / scale[:, 0]
    affine = np.zeros((len(boxes), 2, 3), dtype=np.float32)
    affine[:, 0, 0] = factor
    affine[:, 1, 1] = factor
    affine[:, 0, 2] = np.float32(IMAGE_SIZE * 0.5) - center[:, 0] * factor
    affine[:, 1, 2] = np.float32(IMAGE_SIZE * 0.5) - center[:, 1] * factor
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        raise VithRuntimeError(
            f"OpenCV is required for exact crop warping: {exc}"
        ) from exc
    crops = np.stack(
        [
            cv2.warpAffine(
                image,
                matrix,
                (IMAGE_SIZE, IMAGE_SIZE),
                flags=cv2.INTER_LINEAR,
            )
            for matrix in affine
        ],
        axis=0,
    )
    images = crops.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255.0)
    height, width = image.shape[:2]
    if camera_intrinsics is None:
        focal = np.float32(math.sqrt(height * height + width * width))
        intrinsic = np.asarray(
            ((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
    else:
        intrinsic = np.asarray(camera_intrinsics, dtype=np.float32).reshape(3, 3)
        if (
            not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
            or abs(float(intrinsic[2, 2]) - 1.0) > 1.0e-5
        ):
            raise VithRuntimeError(
                "camera_intrinsics must be a finite pinhole 3x3 matrix"
            )
    return CropBatch(
        images=np.ascontiguousarray(images),
        bbox_center=center,
        bbox_scale=scale.astype(np.float32),
        affine=affine,
        original_size=np.tile(
            np.asarray((width, height), dtype=np.float32), (len(boxes), 1)
        ),
        camera_intrinsics=np.tile(intrinsic[None], (len(boxes), 1, 1)),
    )


def backbone_input(batch: CropBatch) -> np.ndarray:
    normalized = (batch.images - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(
        normalized[..., BACKBONE_TRIM:-BACKBONE_TRIM], dtype=np.float16
    )


def ray_condition(batch: CropBatch) -> np.ndarray:
    coordinates = np.arange(IMAGE_SIZE, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    grid = np.stack((grid_x, grid_y), axis=0)[None]
    diagonal = batch.affine[:, (0, 1), (0, 1)][:, :, None, None]
    translation = batch.affine[:, (0, 1), (2, 2)][:, :, None, None]
    full = grid / diagonal - translation / diagonal
    principal = batch.camera_intrinsics[:, (0, 1), (2, 2)][:, :, None, None]
    focal = batch.camera_intrinsics[:, (0, 1), (0, 1)][:, :, None, None]
    rays = (full - principal) / focal
    return np.ascontiguousarray(
        rays[..., BACKBONE_TRIM:-BACKBONE_TRIM], dtype=np.float32
    )


def decoder_condition(batch: CropBatch) -> np.ndarray:
    principal = batch.camera_intrinsics[:, (0, 1), (2, 2)]
    focal = batch.camera_intrinsics[:, 0, 0:1]
    return np.ascontiguousarray(
        np.concatenate(
            ((batch.bbox_center - principal) / focal, batch.bbox_scale[:, 0:1] / focal),
            axis=1,
        ),
        dtype=np.float32,
    )


def full_to_crop(batch: CropBatch, points: object) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate(
        (values, np.ones((*values.shape[:-1], 1), dtype=np.float32)), axis=-1
    )
    transformed = homogeneous @ np.swapaxes(batch.affine, -1, -2)
    return transformed / np.float32(IMAGE_SIZE) - np.float32(0.5)


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


def project_camera(
    output: dict[str, np.ndarray],
    batch: CropBatch,
    *,
    default_scale_factor: float,
) -> dict[str, np.ndarray]:
    camera = output["pred_cam"].copy()
    camera[:, (0, 2)] *= np.float32(-1.0)
    scale, tx, ty = camera[:, 0], camera[:, 1], camera[:, 2]
    focal = batch.camera_intrinsics[:, 0, 0]
    box_scale = batch.bbox_scale[:, 0]
    denominator = box_scale * scale * np.float32(default_scale_factor) + np.float32(
        1e-8
    )
    depth = np.float32(2.0) * focal / denominator
    correction_x = (
        np.float32(2.0)
        * (batch.bbox_center[:, 0] - batch.camera_intrinsics[:, 0, 2])
        / denominator
    )
    correction_y = (
        np.float32(2.0)
        * (batch.bbox_center[:, 1] - batch.camera_intrinsics[:, 1, 2])
        / denominator
    )
    translation = np.stack(
        (tx + correction_x, ty + correction_y, depth), axis=-1
    ).astype(np.float32)
    camera_points = output["pred_keypoints_3d"] + translation[:, None, :]
    normalized = camera_points / camera_points[..., 2:3]
    projected = np.einsum(
        "bij,bkj->bki", batch.camera_intrinsics, normalized, optimize=True
    )[..., :2]
    output["pred_cam_t"] = translation
    output["focal_length"] = focal.astype(np.float32)
    output["pred_keypoints_2d"] = projected.astype(np.float32)
    output["pred_keypoints_2d_depth"] = camera_points[..., 2].astype(np.float32)
    output["pred_keypoints_2d_cropped"] = full_to_crop(batch, projected)
    return output


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


def hand_boxes_from_tokens(
    normalized_boxes: object, body_batch: CropBatch
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(normalized_boxes, dtype=np.float32) * np.float32(IMAGE_SIZE)
    if boxes.shape != (len(body_batch.images), 2, 4):
        raise VithRuntimeError(f"hand boxes must be [B,2,4], got {boxes.shape}")
    centers = boxes[..., :2]
    sides = np.max(boxes[..., 2:], axis=-1, keepdims=True)
    scales = np.repeat(sides, 2, axis=-1)
    factor = body_batch.affine[:, 0, 0][:, None, None]
    translation = body_batch.affine[:, (0, 1), (2, 2)][:, None, :]
    full_centers = (centers - translation) / factor
    full_scales = scales / factor
    xyxy = np.concatenate(
        (full_centers - full_scales * 0.5, full_centers + full_scales * 0.5),
        axis=-1,
    )
    return np.ascontiguousarray(xyxy[:, 0]), np.ascontiguousarray(xyxy[:, 1])


def _fused_local_wrist(
    predicted_global: np.ndarray, wrist_zero: np.ndarray
) -> np.ndarray:
    # Exact equivalent of upstream ``einsum('kabc,kabd->kadc', ...)``.
    return np.swapaxes(wrist_zero, -1, -2) @ predicted_global


def required_graph_roles() -> tuple[str, ...]:
    roles = ["backbone"]
    for branch in ("body", "hand"):
        roles.append(f"token_initializer.{branch}")
        roles.extend(f"decoder.{branch}.layer.{index}" for index in range(6))
        roles.append(f"decoder.{branch}.final_norm")
        roles.append(f"raw_head.{branch}")
        roles.append(f"keypoint_updater.{branch}")
    roles.append("hand_box_head")
    return tuple(roles)


class GraphSessions:
    """Logical-name adapter around the 22 manifest-declared ORT sessions."""

    def __init__(self, bundle, sessions: Mapping[str, object]) -> None:
        self.bundle = bundle
        self.sessions = dict(sessions)
        declared = {graph.role: graph for graph in bundle.graphs}
        missing = [role for role in required_graph_roles() if role not in declared]
        if missing:
            raise VithRuntimeError(
                "Split bundle is missing graph roles: " + ", ".join(missing)
            )
        extra_sessions = set(self.sessions) - set(declared)
        if extra_sessions:
            raise VithRuntimeError(
                "Sessions were supplied for undeclared roles: "
                + ", ".join(sorted(extra_sessions))
            )
        missing_sessions = set(required_graph_roles()) - set(self.sessions)
        if missing_sessions:
            raise VithRuntimeError(
                "Split graph sessions are missing roles: "
                + ", ".join(sorted(missing_sessions))
            )
        self.graphs = declared
        self.executed_roles: set[str] = set()

    def run(self, role: str, values: Mapping[str, object]) -> dict[str, np.ndarray]:
        graph = self.graphs[role]
        missing = set(graph.inputs) - set(values)
        if missing:
            raise VithRuntimeError(
                f"{role} is missing logical inputs: {', '.join(sorted(missing))}"
            )
        feed = {
            graph.inputs[name]: np.ascontiguousarray(values[name])
            for name in graph.inputs
        }
        logical_outputs = tuple(graph.outputs)
        output_names = [graph.outputs[name] for name in logical_outputs]
        try:
            result = self.sessions[role].run(output_names, feed)
        except Exception as exc:
            raise GraphExecutionError(f"Split graph {role!r} failed: {exc}") from exc
        self.executed_roles.add(role)
        if len(result) != len(logical_outputs):
            raise VithRuntimeError(
                f"Split graph {role!r} returned the wrong output count"
            )
        return {
            name: np.asarray(value)
            for name, value in zip(logical_outputs, result, strict=True)
        }


class VithSplitRunner:
    """Exact six-layer body/hand orchestration over persistent ORT sessions."""

    def __init__(self, bundle, sessions: Mapping[str, object]) -> None:
        self.bundle = bundle
        self.graphs = GraphSessions(bundle, sessions)
        artifacts = {
            PurePosixPath(item.path).name: bundle.root.joinpath(
                *PurePosixPath(item.path).parts
            )
            for item in bundle.artifacts
        }
        required = (
            "body_model.lbs",
            "correctives.bin",
            "keypoint_mapping.bin",
            "hand_runtime.bin",
        )
        missing = [name for name in required if name not in artifacts]
        if missing:
            raise VithRuntimeError(
                "Split bundle is missing MHR artifacts: " + ", ".join(missing)
            )
        self.mhr = MhrModel.load(*(artifacts[name] for name in required))

    def _branch(
        self,
        branch: str,
        features: np.ndarray,
        batch: CropBatch,
        prompt_points: np.ndarray,
        prompt_labels: np.ndarray,
        prev_estimate: np.ndarray,
        *,
        use_initial_estimate: bool,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        people = len(features)
        initialized = self.graphs.run(
            f"token_initializer.{branch}",
            {
                "features": np.ascontiguousarray(features, dtype=np.float32),
                "condition_info": decoder_condition(batch),
                "ray_cond": ray_condition(batch),
                "prompt_points": np.ascontiguousarray(prompt_points, dtype=np.float32),
                "prompt_labels": np.ascontiguousarray(prompt_labels, dtype=np.int64),
                "prev_estimate": np.ascontiguousarray(prev_estimate, dtype=np.float32),
                "use_initial_estimate": np.full(
                    people, int(use_initial_estimate), dtype=np.int64
                ),
            },
        )
        tokens = np.ascontiguousarray(initialized["tokens"], dtype=np.float32)
        token_pe = np.ascontiguousarray(initialized["token_pe"], dtype=np.float32)
        image_features = np.ascontiguousarray(
            initialized["image_features"], dtype=np.float32
        )
        image_pe_features = np.ascontiguousarray(
            initialized["image_pe"], dtype=np.float32
        )
        image_tokens = np.ascontiguousarray(
            image_features.transpose(0, 2, 3, 1).reshape(people, 32 * 24, 1280)
        )
        image_pe = np.ascontiguousarray(
            image_pe_features.transpose(0, 2, 3, 1).reshape(people, 32 * 24, 1280)
        )
        output = None
        normalized = None
        for index in range(6):
            tokens = self.graphs.run(
                f"decoder.{branch}.layer.{index}",
                {
                    "tokens": tokens,
                    "image_tokens": image_tokens,
                    "token_pe": token_pe,
                    "image_pe": image_pe,
                },
            )["tokens_out"]
            normalized = self.graphs.run(
                f"decoder.{branch}.final_norm", {"tokens": tokens}
            )["normalized_tokens"]
            head = self.graphs.run(
                f"raw_head.{branch}", {"pose_token": normalized[:, 0]}
            )
            output = decode_raw_mhr(
                self.mhr,
                head["raw_mhr_parameters"],
                head["raw_camera"],
                hand_model=branch == "hand",
            )
            project_camera(
                output,
                batch,
                default_scale_factor=10.0 if branch == "hand" else 1.0,
            )
            if index < 5:
                updated = self.graphs.run(
                    f"keypoint_updater.{branch}",
                    {
                        "tokens": tokens,
                        "token_pe": token_pe,
                        "image_features": image_features,
                        "keypoints_2d_crop": output["pred_keypoints_2d_cropped"][
                            :, :70
                        ],
                        "keypoint_depth": output["pred_keypoints_2d_depth"][:, :70],
                        "keypoints_3d": output["pred_keypoints_3d"][:, :70],
                    },
                )
                tokens = updated["updated_tokens"]
                token_pe = updated["updated_token_pe"]
        if output is None or normalized is None:
            raise AssertionError("six-layer decoder produced no output")
        return output, normalized

    def _backbone(self, batch: CropBatch) -> np.ndarray:
        return np.asarray(
            self.graphs.run("backbone", {"image": backbone_input(batch)})["features"],
            dtype=np.float32,
        )

    def _initial_branch(
        self, branch: str, batch: CropBatch
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        features = self._backbone(batch)
        people = len(features)
        points = np.zeros((people, 1, 2), dtype=np.float32)
        labels = np.full((people, 1), -2, dtype=np.int64)
        prev = np.zeros((people, 522), dtype=np.float32)
        output, normalized = self._branch(
            branch,
            features,
            batch,
            points,
            labels,
            prev,
            use_initial_estimate=True,
        )
        return output, normalized, features

    def _body_only(self, body_batch: CropBatch) -> dict[str, np.ndarray]:
        body, _, _ = self._initial_branch("body", body_batch)
        return {
            "pred_keypoints_3d": np.ascontiguousarray(body["pred_keypoints_3d"]),
            "pred_keypoints_2d": np.ascontiguousarray(body["pred_keypoints_2d"]),
            "pred_cam_t": np.ascontiguousarray(body["pred_cam_t"]),
            "focal_length": np.ascontiguousarray(body["focal_length"]),
            "global_rot": np.ascontiguousarray(body["global_rot"]),
            "scale_params": np.ascontiguousarray(body["scale"]),
            "pred_joint_coords": np.ascontiguousarray(body["pred_joint_coords"]),
            "pred_global_rots": np.ascontiguousarray(body["joint_global_rots"]),
        }

    def _full_hand(
        self, image_rgb: np.ndarray, body_batch: CropBatch
    ) -> dict[str, np.ndarray]:
        people = len(body_batch.images)
        height, width = image_rgb.shape[:2]
        body, body_tokens, body_features = self._initial_branch("body", body_batch)
        # The initial pass has one prompt.  Pose/previous/prompt occupy tokens
        # 0..2, followed by the two hand detector tokens.
        hand_box = self.graphs.run(
            "hand_box_head", {"hand_tokens": body_tokens[:, 3:5]}
        )["hand_boxes"]
        left_xyxy, right_xyxy = hand_boxes_from_tokens(hand_box, body_batch)
        wrist_pose_indices = np.asarray((41, 43, 42, 31, 33, 32), dtype=np.intp)
        original_local_wrist = euler_xzy_to_matrix(
            body["body_pose"][:, wrist_pose_indices].reshape(people, 2, 3)
        )

        flipped = np.ascontiguousarray(image_rgb[:, ::-1])
        flipped_left = left_xyxy.copy()
        flipped_left[:, 0] = width - left_xyxy[:, 2] - 1
        flipped_left[:, 2] = width - left_xyxy[:, 0] - 1
        right_intrinsic = body_batch.camera_intrinsics[0]
        left_intrinsic = right_intrinsic.copy()
        left_intrinsic[0, 2] = np.float32(width - 1) - left_intrinsic[0, 2]
        left_batch = prepare_crop_batch(
            flipped,
            flipped_left,
            padding=0.9,
            camera_intrinsics=left_intrinsic,
        )
        right_batch = prepare_crop_batch(
            image_rgb,
            right_xyxy,
            padding=0.9,
            camera_intrinsics=right_intrinsic,
        )
        left, _, _ = self._initial_branch("hand", left_batch)
        right, _, _ = self._initial_branch("hand", right_batch)

        mean_right = self.mhr.scale_mean[8]
        mean_left = self.mhr.scale_mean[9]
        component_right = self.mhr.scale_components[8, 8]
        component_left = self.mhr.scale_components[9, 9]
        left["scale"][:, 9] = (
            mean_right + component_right * left["scale"][:, 8] - mean_left
        ) / component_left
        left["joint_global_rots"][:, 78] = left["joint_global_rots"][:, 42]
        left["joint_global_rots"][:, 78, (1, 2), :] *= np.float32(-1.0)
        left["hand"][:, :54] = left["hand"][:, 54:]
        # Match the upstream batch mutation after hand decoding.  The already
        # computed crop projection remains in flipped-image coordinates.
        left_batch.bbox_center[:, 0] = width - left_batch.bbox_center[:, 0] - 1

        lowarm = body["joint_global_rots"][:, (76, 40)]
        wrist_zero = lowarm @ self.mhr.joint_rotation[[77, 41]]
        predicted_wrist = np.stack(
            (left["joint_global_rots"][:, 78], right["joint_global_rots"][:, 42]),
            axis=1,
        )
        fused_local = _fused_local_wrist(predicted_wrist, wrist_zero)
        valid_angle_initial = (
            rotation_angle_difference(original_local_wrist, fused_local) < 1.4
        )
        valid_box = np.stack(
            (
                np.all(left_batch.bbox_scale > 64.0, axis=1),
                np.all(right_batch.bbox_scale > 64.0, axis=1),
            ),
            axis=1,
        )
        valid_inside = np.stack(
            (
                np.max(np.abs(left["pred_keypoints_2d_cropped"]), axis=(1, 2)) < 0.5,
                np.max(np.abs(right["pred_keypoints_2d_cropped"]), axis=(1, 2)) < 0.5,
            ),
            axis=1,
        )
        right_wrist = right["pred_keypoints_2d"][:, [41]].copy()
        left_wrist = left["pred_keypoints_2d"][:, [41]].copy()
        left_wrist[:, :, 0] = width - left_wrist[:, :, 0] - 1
        body_right_wrist = body["pred_keypoints_2d"][:, [41]]
        body_left_wrist = body["pred_keypoints_2d"][:, [62]]
        # Preserve the pinned implementation's intentionally crossed scale
        # denominators and left/right stack order.
        right_distance = (
            np.linalg.norm((right_wrist - body_right_wrist).reshape(people, 2), axis=1)
            / left_batch.bbox_scale[:, 0]
        )
        left_distance = (
            np.linalg.norm((left_wrist - body_left_wrist).reshape(people, 2), axis=1)
            / right_batch.bbox_scale[:, 0]
        )
        valid_wrist_distance = np.stack(
            (left_distance < 0.25, right_distance < 0.25), axis=1
        )
        hand_valid = (
            valid_angle_initial & valid_box & valid_inside & valid_wrist_distance
        )

        right_wrist_crop = full_to_crop(body_batch, right_wrist)
        left_wrist_crop = full_to_crop(body_batch, left_wrist)
        right_elbow_crop = full_to_crop(body_batch, body["pred_keypoints_2d"][:, [8]])
        left_elbow_crop = full_to_crop(body_batch, body["pred_keypoints_2d"][:, [7]])
        prompt_points = np.concatenate(
            (right_wrist_crop, left_wrist_crop, right_elbow_crop, left_elbow_crop),
            axis=1,
        )
        prompt_labels = np.tile(np.asarray((41, 62, 8, 7), dtype=np.int64), (people, 1))
        prompt_hand_valid = hand_valid[:, (1, 0, 1, 0)]
        if people > 1:
            invalid = (
                (prompt_points[..., 0] < -0.5)
                | (prompt_points[..., 0] > 0.5)
                | (prompt_points[..., 1] < -0.5)
                | (prompt_points[..., 1] > 0.5)
                | ~prompt_hand_valid
            )
            prompt_points = np.clip(prompt_points + 0.5, 0.0, 1.0)
            prompt_points[invalid] = 0.0
            prompt_labels[invalid] = -2
        else:
            valid = (
                np.all((prompt_points[0] > -0.5) & (prompt_points[0] < 0.5), axis=1)
                & prompt_hand_valid[0]
            )
            prompt_points = np.clip(prompt_points[:, valid] + 0.5, 0.0, 1.0)
            prompt_labels = prompt_labels[:, valid]
        if prompt_points.shape[1]:
            body, _ = self._branch(
                "body",
                body_features,
                body_batch,
                prompt_points,
                prompt_labels,
                previous_estimate(body),
                use_initial_estimate=False,
            )

        updated_hand = np.concatenate(
            (left["hand"][:, :54], right["hand"][:, 54:]), axis=1
        )
        updated_scale = body["scale"].copy()
        updated_scale[:, 9] = left["scale"][:, 9]
        updated_scale[:, 8] = right["scale"][:, 8]
        updated_scale[:, 18:] = (left["scale"][:, 18:] + right["scale"][:, 18:]) * 0.5
        updated_shape = body["shape"].copy()
        updated_shape[:, 40:] = (left["shape"][:, 40:] + right["shape"][:, 40:]) * 0.5

        fk_parameters, _ = _mhr_model_parameters(
            self.mhr,
            global_rotation=body["global_rot"],
            body_pose=body["body_pose"],
            hand_pose=updated_hand,
            scale=updated_scale,
            hand_model=False,
        )
        fk_rotations = self.mhr.forward(
            updated_shape, fk_parameters, body["face"]
        ).joint_rotations
        lowarm = fk_rotations[:, (76, 40)]
        wrist_zero = lowarm @ self.mhr.joint_rotation[[77, 41]]
        fused_local = _fused_local_wrist(predicted_wrist, wrist_zero)
        wrist_xzy = fix_wrist_euler(matrix_to_euler_xzy(fused_local))
        valid_angle = (
            rotation_angle_difference(original_local_wrist, fused_local) < 1.4
        ) & hand_valid

        original_body_wrist = body["body_pose"][:, wrist_pose_indices].reshape(
            people, 2, 3
        )
        body["body_pose"][:, wrist_pose_indices] = np.where(
            valid_angle[..., None], wrist_xzy, original_body_wrist
        ).reshape(people, 6)
        body_hand = body["hand"].reshape(people, 2, 54)
        body["hand"] = np.where(
            valid_angle[..., None], updated_hand.reshape(people, 2, 54), body_hand
        ).reshape(people, 108)
        original_hand_scale = np.stack(
            (body["scale"][:, 9], body["scale"][:, 8]), axis=1
        )
        replacement_hand_scale = np.stack(
            (updated_scale[:, 9], updated_scale[:, 8]), axis=1
        )
        selected_hand_scale = np.where(
            valid_angle, replacement_hand_scale, original_hand_scale
        )
        body["scale"][:, 9] = selected_hand_scale[:, 0]
        body["scale"][:, 8] = selected_hand_scale[:, 1]
        valid_count = valid_angle.sum(axis=1, keepdims=True).astype(np.float32)
        has_valid = valid_count > 0
        shared_scale = (
            left["scale"][:, 18:] * valid_angle[:, [0]]
            + right["scale"][:, 18:] * valid_angle[:, [1]]
        ) / (valid_count + np.float32(1e-8))
        shared_shape = (
            left["shape"][:, 40:] * valid_angle[:, [0]]
            + right["shape"][:, 40:] * valid_angle[:, [1]]
        ) / (valid_count + np.float32(1e-8))
        body["scale"][:, 18:] = np.where(has_valid, shared_scale, body["scale"][:, 18:])
        body["shape"][:, 40:] = np.where(has_valid, shared_shape, body["shape"][:, 40:])

        final_parameters, _ = _mhr_model_parameters(
            self.mhr,
            global_rotation=body["global_rot"],
            body_pose=body["body_pose"],
            hand_pose=body["hand"],
            scale=body["scale"],
            hand_model=False,
        )
        final_geometry = self.mhr.forward(
            body["shape"], final_parameters, body["face"]
        ).camera_space()
        body["pred_keypoints_3d"] = final_geometry.keypoints
        body["pred_joint_coords"] = final_geometry.joint_coords
        # Preserve the pinned upstream result contract.  The hand-refinement
        # path recomputes geometry after wrist/hand fusion, but it never writes
        # the recomputed joint rotations back to ``pose_output``.  Therefore
        # ``pred_global_rots`` intentionally remains the decoder result from
        # the prompted body pass (or initial pass when there are no prompts).
        body["mhr_model_params"] = final_parameters
        camera_points = body["pred_keypoints_3d"] + body["pred_cam_t"][:, None, :]
        normalized = camera_points / camera_points[..., 2:3]
        body["pred_keypoints_2d"] = np.einsum(
            "bij,bkj->bki",
            body_batch.camera_intrinsics,
            normalized,
            optimize=True,
        )[..., :2]
        return {
            "pred_keypoints_3d": np.ascontiguousarray(body["pred_keypoints_3d"]),
            "pred_keypoints_2d": np.ascontiguousarray(body["pred_keypoints_2d"]),
            "pred_cam_t": np.ascontiguousarray(body["pred_cam_t"]),
            "focal_length": np.ascontiguousarray(body["focal_length"]),
            "global_rot": np.ascontiguousarray(body["global_rot"]),
            "scale_params": np.ascontiguousarray(body["scale"]),
            "pred_joint_coords": np.ascontiguousarray(body["pred_joint_coords"]),
            "pred_global_rots": np.ascontiguousarray(body["joint_global_rots"]),
        }

    def run(
        self,
        image_rgb: object,
        bboxes_xyxy: object,
        *,
        include_hands: bool,
        camera_intrinsics: object | None = None,
    ) -> dict[str, np.ndarray]:
        self.graphs.executed_roles.clear()
        image = np.asarray(image_rgb)
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return {
                "pred_keypoints_3d": np.empty((0, 70, 3), dtype=np.float32),
                "pred_keypoints_2d": np.empty((0, 70, 2), dtype=np.float32),
                "pred_cam_t": np.empty((0, 3), dtype=np.float32),
                "focal_length": np.empty((0,), dtype=np.float32),
                "global_rot": np.empty((0, 3), dtype=np.float32),
                "scale_params": np.empty((0, 28), dtype=np.float32),
                "pred_joint_coords": np.empty((0, 127, 3), dtype=np.float32),
                "pred_global_rots": np.empty((0, 127, 3, 3), dtype=np.float32),
            }
        body_batch = prepare_crop_batch(
            image,
            boxes,
            padding=1.25,
            camera_intrinsics=camera_intrinsics,
        )
        return (
            self._full_hand(image, body_batch)
            if include_hands
            else self._body_only(body_batch)
        )

    def close(self) -> None:
        mhr = getattr(self, "mhr", None)
        if mhr is not None:
            mhr.close()
            self.mhr = None
        self.graphs.sessions.clear()
        gc.collect()


class VithEstimator:
    """Plugin-facing persistent split estimator with EP/device verification."""

    def __init__(
        self,
        bundle,
        requested_mode: object,
        *,
        ort_module=None,
        session_factory=None,
        sys_platform: str | None = None,
        verify_accelerator_execution: bool = False,
    ) -> None:
        runtime = _ort_runtime
        if bundle.runner != runtime.SPLIT_NUMPY_RUNNER:
            raise VithRuntimeError("VithEstimator requires onnxruntime.split_numpy.v1")
        self.bundle = bundle
        self.requested_mode = runtime.normalize_mode(requested_mode)
        self._ort = ort_module or runtime.import_onnxruntime()
        runtime._register_webgpu_plugin(self._ort)
        available = runtime.available_execution_providers(self._ort)
        self._available_providers = tuple(available)
        selected, device, fallback = runtime.select_execution_provider(
            self.requested_mode, available, sys_platform=sys_platform
        )
        cpu_available = any(
            name.casefold() == "cpuexecutionprovider" for name in available
        )
        factory = session_factory or self._ort.InferenceSession
        self._session_factory = factory
        hardware = runtime._hardware_device(self._ort, selected)
        self._profile_requested = bool(
            verify_accelerator_execution
            and selected.casefold() != "cpuexecutionprovider"
        )
        self._profile_failure: VithRuntimeError | None = None
        self.accelerator_profile_evidence: dict[str, object] = {}
        self._profile_prefixes: dict[str, Path] = {}
        self._runner = None
        self._sessions: dict[str, object] = {}
        try:
            self._sessions = self._create_sessions(
                bundle,
                factory,
                selected,
                available,
                hardware,
                profile=self._profile_requested,
            )
        except Exception as exc:
            self._sessions.clear()
            if (
                self.requested_mode == "AUTO"
                and selected.casefold() != "cpuexecutionprovider"
                and cpu_available
            ):
                try:
                    self._sessions = self._create_sessions(
                        bundle,
                        factory,
                        "CPUExecutionProvider",
                        available,
                        None,
                        profile=False,
                    )
                except Exception as cpu_exc:
                    raise VithRuntimeError(
                        "Could not create all split SAM sessions with "
                        f"{selected} or CPU fallback: {cpu_exc}"
                    ) from cpu_exc
                selected, device, fallback = "CPUExecutionProvider", "cpu", True
                hardware = None
                self._profile_requested = False
            else:
                raise VithRuntimeError(
                    f"Could not create all split SAM sessions with {selected}: {exc}"
                ) from exc
        actual_providers = self._actual_session_providers()
        unexpected = {
            provider
            for provider in actual_providers
            if provider.casefold() != selected.casefold()
        }
        if unexpected:
            cpu_only = all(
                provider.casefold() == "cpuexecutionprovider" for provider in unexpected
            )
            if self.requested_mode == "GPU" or not cpu_only:
                self.close()
                raise VithRuntimeError(
                    "Split sessions selected unexpected providers: "
                    + ", ".join(sorted(unexpected))
                )
            self._dispose_sessions(self._sessions, profiled=self._profile_requested)
            try:
                self._sessions = self._create_sessions(
                    bundle,
                    factory,
                    "CPUExecutionProvider",
                    available,
                    None,
                    profile=False,
                )
            except Exception as cpu_exc:
                raise VithRuntimeError(
                    "Could not recreate every split SAM session on the CPU "
                    f"after accelerator fallback: {cpu_exc}"
                ) from cpu_exc
            selected, device, fallback = "CPUExecutionProvider", "cpu", True
            hardware = None
            self._profile_requested = False
            if not self._all_sessions_use("CPUExecutionProvider"):
                self._dispose_sessions(self._sessions, profiled=False)
                raise VithRuntimeError(
                    "CPU fallback did not recreate every split session exclusively "
                    "on CPUExecutionProvider"
                )
        vendor_id, device_id = runtime._hardware_ids(hardware)
        self.diagnostics = runtime.RuntimeDiagnostics(
            requested_mode=self.requested_mode,
            actual_ep=selected,
            actual_device=device,
            fallback=fallback,
            model_revision=bundle.model_revision,
            precision=bundle.precision,
            available_providers=available,
            hardware_vendor_id=vendor_id,
            hardware_device_id=device_id,
        )
        self.neutral_rotations = dict(bundle.neutral_rotations)
        try:
            self._runner = VithSplitRunner(bundle, self._sessions)
        except Exception as exc:
            self._dispose_sessions(
                self._sessions, profiled=bool(self._profile_requested)
            )
            self._profile_requested = False
            self._ort = None
            if isinstance(exc, VithRuntimeError):
                raise
            raise VithRuntimeError(
                f"Could not initialize the split NumPy orchestrator: {exc}"
            ) from exc

    def _actual_session_providers(self) -> list[str]:
        actual: list[str] = []
        for session in self._sessions.values():
            try:
                providers = tuple(session.get_providers())
            except (AttributeError, RuntimeError, TypeError):
                providers = ()
            if providers:
                actual.append(str(providers[0]))
        return actual

    def _all_sessions_use(self, provider: str) -> bool:
        actual = self._actual_session_providers()
        return len(actual) == len(self._sessions) and all(
            value.casefold() == provider.casefold() for value in actual
        )

    def _delete_profile_files(self) -> None:
        for prefix in tuple(self._profile_prefixes.values()):
            for candidate in prefix.parent.glob(prefix.name + "*"):
                if candidate.is_file():
                    candidate.unlink(missing_ok=True)
        self._profile_prefixes.clear()

    def _dispose_sessions(self, sessions: dict[str, object], *, profiled: bool) -> None:
        for session in tuple(sessions.values()):
            profile_path = None
            if profiled:
                try:
                    profile_path = Path(session.end_profiling())
                except Exception:  # ONNX Runtime raises its own exception types.
                    LOGGER.exception("Could not end ONNX Runtime profiling")
                finally:
                    if profile_path is not None and profile_path.is_file():
                        profile_path.unlink(missing_ok=True)
        self._delete_profile_files()
        sessions.clear()
        gc.collect()

    def _create_sessions(
        self,
        bundle,
        factory,
        selected: str,
        available: Sequence[str],
        hardware,
        *,
        profile: bool,
    ) -> dict[str, object]:
        providers = [selected]
        if selected.casefold() != "cpuexecutionprovider" and any(
            name.casefold() == "cpuexecutionprovider" for name in available
        ):
            providers.append("CPUExecutionProvider")
        sessions: dict[str, object] = {}
        options_type = getattr(self._ort, "SessionOptions", None)
        try:
            for ordinal, graph in enumerate(bundle.graphs):
                options = options_type() if options_type is not None else None
                if profile:
                    if options is None:
                        raise VithRuntimeError(
                            "This ORT build cannot profile split accelerator sessions"
                        )
                    options.enable_profiling = True
                    prefix = Path(tempfile.gettempdir()) / (
                        f"mtp-sam3d-vith-{os.getpid()}-"
                        f"{threading.get_ident()}-{ordinal}"
                    )
                    options.profile_file_prefix = str(prefix)
                    self._profile_prefixes[graph.role] = prefix
                path = str(bundle.graph_path_for_role(graph.role))
                add_devices = (
                    getattr(options, "add_provider_for_devices", None)
                    if options is not None
                    else None
                )
                if hardware is not None and callable(add_devices):
                    add_devices(
                        [hardware],
                        {"preferredLayout": "NHWC", "enableGraphCapture": "0"},
                    )
                    sessions[graph.role] = factory(path, sess_options=options)
                elif options is not None:
                    sessions[graph.role] = factory(
                        path, sess_options=options, providers=providers
                    )
                else:
                    sessions[graph.role] = factory(path, providers=providers)
        except Exception:
            self._dispose_sessions(sessions, profiled=profile)
            raise
        return sessions

    def _verify_accelerator_profiles(self, *, full_inference: bool) -> None:
        if self._profile_failure is not None:
            raise self._profile_failure
        if not self._profile_requested or not full_inference:
            # A body-only request may run while profiling is active, but it can
            # never certify the release's full-hand 22-graph execution path.
            return

        expected_roles = set(required_graph_roles())
        executed_roles = set(self._runner.graphs.executed_roles)
        role_counts: dict[str, dict[str, object]] = {}
        errors: list[str] = []
        actual = self.diagnostics.actual_ep.casefold()
        if not any(name in actual for name in ("webgpu", "coreml")):
            errors.append(
                "selected provider is not a supported WebGPU/CoreML accelerator"
            )
        if executed_roles != expected_roles:
            missing = sorted(expected_roles - executed_roles)
            extra = sorted(executed_roles - expected_roles)
            errors.append(
                "full inference graph execution mismatch "
                f"(missing={missing}, extra={extra})"
            )

        for role, session in self._sessions.items():
            profile_path = None
            providers: list[str] = []
            try:
                profile_path = Path(session.end_profiling())
                payload = json.loads(profile_path.read_text(encoding="utf8"))
                if not isinstance(payload, list):
                    raise ValueError("profile root is not an event array")
                providers = [
                    str(item.get("args", {}).get("provider", ""))
                    for item in payload
                    if isinstance(item, dict)
                    and isinstance(item.get("args"), dict)
                    and item.get("args", {}).get("provider")
                ]
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append(f"{role}: could not read profile: {exc}")
            finally:
                if profile_path is not None and profile_path.is_file():
                    profile_path.unlink(missing_ok=True)
            provider_counts: dict[str, int] = {}
            for provider in providers:
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            accelerated = sum(provider.casefold() == actual for provider in providers)
            role_counts[role] = {
                "accelerator_nodes": accelerated,
                "profiled_nodes": len(providers),
                "providers": dict(sorted(provider_counts.items())),
            }
            if role in expected_roles and accelerated < 1:
                errors.append(f"{role}: no compute executed on the accelerator")
        self._profile_requested = False
        self._delete_profile_files()

        accelerated = sum(
            int(value["accelerator_nodes"]) for value in role_counts.values()
        )
        total = sum(int(value["profiled_nodes"]) for value in role_counts.values())
        fraction = accelerated / max(total, 1)
        minimum = 32 if "webgpu" in actual else 1
        minimum_fraction = 0.25 if "webgpu" in actual else 0.0
        if accelerated < minimum or fraction < minimum_fraction:
            errors.append(
                "aggregate accelerator execution was insufficient: "
                f"{accelerated}/{total} nodes"
            )

        self.accelerator_profile_evidence = {
            "passed": not errors,
            "provider": self.diagnostics.actual_ep,
            "full_inference": True,
            "executed_roles": sorted(executed_roles),
            "accelerator_nodes": accelerated,
            "profiled_nodes": total,
            "accelerator_fraction": fraction,
            "roles": role_counts,
        }
        if errors:
            failure = VithRuntimeError(
                "Could not verify full split accelerator execution: "
                + "; ".join(errors)
            )
            self._profile_failure = failure
            self._dispose_sessions(self._sessions, profiled=False)
            poisoned_runner = self._runner
            self._runner = None
            if poisoned_runner is not None:
                poisoned_runner.close()
            raise failure

        from dataclasses import replace

        self.diagnostics = replace(
            self.diagnostics,
            accelerator_nodes_verified=True,
            accelerator_node_count=accelerated,
            profiled_node_count=total,
        )

    def _can_fallback_after_inference_error(self) -> bool:
        return (
            self.requested_mode == "AUTO"
            and self.diagnostics.actual_ep.casefold() != "cpuexecutionprovider"
            and any(
                provider.casefold() == "cpuexecutionprovider"
                for provider in self._available_providers
            )
        )

    def _fallback_to_cpu(self, accelerator_error: Exception) -> None:
        old_runner = self._runner
        self._runner = None
        if old_runner is not None:
            old_runner.close()
        self._dispose_sessions(self._sessions, profiled=bool(self._profile_requested))
        self._profile_requested = False
        self._profile_failure = None
        try:
            self._sessions = self._create_sessions(
                self.bundle,
                self._session_factory,
                "CPUExecutionProvider",
                self._available_providers,
                None,
                profile=False,
            )
            if not self._all_sessions_use("CPUExecutionProvider"):
                raise VithRuntimeError(
                    "not every recreated session selected CPUExecutionProvider"
                )
            self._runner = VithSplitRunner(self.bundle, self._sessions)
        except Exception as cpu_exc:
            self._dispose_sessions(self._sessions, profiled=False)
            raise VithRuntimeError(
                "Accelerator inference failed and the complete CPU retry could "
                f"not be initialized: {cpu_exc}"
            ) from cpu_exc

        from dataclasses import replace

        self.diagnostics = replace(
            self.diagnostics,
            actual_ep="CPUExecutionProvider",
            actual_device="cpu",
            fallback=True,
            hardware_vendor_id=0,
            hardware_device_id=0,
            accelerator_nodes_verified=False,
            accelerator_node_count=0,
            profiled_node_count=0,
        )
        self.accelerator_profile_evidence = {
            "passed": False,
            "fallback": "CPUExecutionProvider",
            "accelerator_error": str(accelerator_error),
        }

    def _run_output(
        self,
        image_rgb: object,
        boxes: np.ndarray,
        *,
        include_hands: bool,
        camera_intrinsics: object | None = None,
    ) -> dict[str, np.ndarray]:
        if self._profile_failure is not None:
            raise self._profile_failure
        try:
            output = self._runner.run(
                image_rgb,
                boxes,
                include_hands=include_hands,
                camera_intrinsics=camera_intrinsics,
            )
        except GraphExecutionError as exc:
            if self._can_fallback_after_inference_error():
                self._fallback_to_cpu(exc)
                output = self._runner.run(
                    image_rgb,
                    boxes,
                    include_hands=include_hands,
                    camera_intrinsics=camera_intrinsics,
                )
            else:
                if self._profile_requested:
                    self._dispose_sessions(self._sessions, profiled=True)
                    self._profile_requested = False
                    poisoned_runner = self._runner
                    self._runner = None
                    if poisoned_runner is not None:
                        poisoned_runner.close()
                    failure = VithRuntimeError(
                        "Accelerator inference failed before execution could be "
                        f"verified: {exc}"
                    )
                    self._profile_failure = failure
                    raise failure from exc
                raise
        _ort_runtime._validate_output_tensors(np, output, len(boxes))
        self._verify_accelerator_profiles(full_inference=include_hands)
        return output

    def process_one_image(
        self,
        image_rgb,
        *,
        bboxes,
        inference_type: str = "full",
        camera_intrinsics=None,
    ) -> list[dict[str, np.ndarray]]:
        boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return []
        if inference_type not in {"full", "body"}:
            raise VithRuntimeError(
                "Split estimator inference_type must be 'full' or 'body'"
            )
        output = self._run_output(
            image_rgb,
            boxes,
            include_hands=inference_type == "full",
            camera_intrinsics=camera_intrinsics,
        )
        return [
            {key: output[key][index] for key in OUTPUT_KEYS}
            for index in range(len(boxes))
        ]

    def close(self) -> None:
        # End active profiles solely to obtain and delete their files.  Close
        # must never leave potentially identifying temp paths behind.
        sessions = getattr(self, "_sessions", {})
        self._dispose_sessions(
            sessions,
            profiled=bool(getattr(self, "_profile_requested", False)),
        )
        self._profile_requested = False
        runner = getattr(self, "_runner", None)
        if runner is not None:
            runner.close()
        self._runner = None
        self._sessions = {}
        self._ort = None
        gc.collect()


def run_bundle(
    *,
    manifest_path: str | Path,
    image_rgb: object,
    bboxes_xyxy: object,
    include_hands: bool,
    providers: Sequence[str] | None,
    camera_intrinsics: object | None = None,
) -> dict[str, np.ndarray]:
    """One-shot publication/parity entry point with the stable 8-output API."""

    runtime = _ort_runtime
    bundle = runtime.load_bundle_manifest(manifest_path, verify_files=True)
    requested = "CPU"
    if providers and any(
        str(provider).casefold() != "cpuexecutionprovider" for provider in providers
    ):
        requested = "GPU"
    estimator = VithEstimator(
        bundle,
        requested,
        verify_accelerator_execution=requested == "GPU",
    )
    try:
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        return estimator._run_output(
            image_rgb,
            boxes,
            include_hands=bool(include_hands),
            camera_intrinsics=camera_intrinsics,
        )
    finally:
        estimator.close()


__all__ = [
    "CropBatch",
    "GraphExecutionError",
    "GraphSessions",
    "VithEstimator",
    "VithRuntimeError",
    "VithSplitRunner",
    "backbone_input",
    "compact_cont_to_body",
    "compact_cont_to_hand",
    "decode_raw_mhr",
    "decoder_condition",
    "fix_wrist_euler",
    "full_to_crop",
    "hand_boxes_from_tokens",
    "prepare_crop_batch",
    "project_camera",
    "ray_condition",
    "required_graph_roles",
    "rotation_6d_to_matrix",
    "run_bundle",
]
