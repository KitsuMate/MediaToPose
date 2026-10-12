"""NumPy orchestrator for the split SAM 3D Body ViT-H ONNX bundle.

The learned model is kept in 22 small, independently verifiable ONNX graphs
(one backbone plus 21 decoder components). Crop geometry, camera projection,
MHR decoding and body/hand fusion are deterministic NumPy operations, with no
PyTorch, PyMomentum, Roma or exporter dependency.
"""

from __future__ import annotations

from .common import GraphExecutionError, VithRuntimeError
from .crops import (
    CropBatch,
    backbone_input,
    decoder_condition,
    full_to_crop,
    hand_boxes_from_tokens,
    prepare_crop_batch,
    project_camera,
    ray_condition,
)
from .estimator import VithEstimator, run_bundle
from .graphs import GraphSessions, required_graph_roles
from .mhr_params import compact_cont_to_body, compact_cont_to_hand, decode_raw_mhr
from .rotations import fix_wrist_euler, rotation_6d_to_matrix
from .runner import VithSplitRunner

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
