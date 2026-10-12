"""Input geometry constants, output keys and errors of the ViT-H bundle."""

from __future__ import annotations

import numpy as np

from .. import ort_runtime as _ort_runtime

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


class VithRuntimeError(_ort_runtime.OrtRuntimeError):
    """The split bundle or a deterministic runtime input is incompatible."""


class GraphExecutionError(VithRuntimeError):
    """An ORT component session failed while executing a declared graph."""
