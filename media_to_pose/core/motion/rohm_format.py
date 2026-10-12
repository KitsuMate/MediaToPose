"""The RoHM motion representation: frame rate, window sizes and feature layout.

Shared by the ONNX runtime that runs the model and by the model-free code
that builds and reads its 294-feature frames.
"""

from __future__ import annotations

import numpy as np

from ..errors import BackendError

MODEL_FPS = 30.0
SOURCE_WINDOW_FRAMES = 145
MODEL_WINDOW_FRAMES = 144
FEATURE_COUNT = 294
TRAJECTORY_COUNT = 22
POSE_COUNT = 272
TRAJECTORY_ABSOLUTE_INDICES = np.asarray(
    (0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18), dtype=np.int64
)
POSE_START = 22
BETAS_SLICE = slice(280, 290)
FOOT_CONTACT_SLICE = slice(290, 294)


def visibility_vector(joint_visibility: np.ndarray) -> np.ndarray:
    """Expand per-joint visibility [T, 22] to the 294-feature mask."""
    visible = np.asarray(joint_visibility, dtype=bool)
    if visible.ndim != 2 or visible.shape[1] != 22:
        raise BackendError("RoHM visibility must have shape [T, 22]")
    result = np.ones((len(visible), FEATURE_COUNT), dtype=np.float32)
    result[:, 22:88] = np.repeat(visible, 3, axis=1)
    result[:, 88:154] = np.repeat(visible, 3, axis=1)
    result[:, 154:280] = np.repeat(visible[:, 1:], 6, axis=1)
    result[:, FOOT_CONTACT_SLICE] = 0.0
    left = visible[:, 7] & visible[:, 10]
    right = visible[:, 8] & visible[:, 11]
    result[left, 290:292] = 1.0
    result[right, 292:294] = 1.0
    return result


__all__ = [
    "BETAS_SLICE",
    "FEATURE_COUNT",
    "FOOT_CONTACT_SLICE",
    "MODEL_FPS",
    "MODEL_WINDOW_FRAMES",
    "POSE_COUNT",
    "POSE_START",
    "SOURCE_WINDOW_FRAMES",
    "TRAJECTORY_ABSOLUTE_INDICES",
    "TRAJECTORY_COUNT",
    "visibility_vector",
]
