"""Person and hand crops, backbone inputs and camera projection."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np

from .common import BACKBONE_TRIM, IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, VithRuntimeError


@dataclass(slots=True)
class CropBatch:
    images: np.ndarray
    bbox_center: np.ndarray
    bbox_scale: np.ndarray
    affine: np.ndarray
    original_size: np.ndarray
    camera_intrinsics: np.ndarray


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
