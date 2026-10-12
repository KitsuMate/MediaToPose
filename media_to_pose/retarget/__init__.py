"""Convert canonical detections into Blender pose, shape-key and keyframe channels.

The public entry points are re-exported lazily: importing one submodule
(for example ``retarget.target_pose`` from ``postprocess``) must not import
the whole package, because ``retarget.pose`` itself depends on
``postprocess``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "SequenceApplyProgress": ".samples",
    "apply_frame": ".pose",
    "apply_sequence": ".sequence",
    "body_points": ".solve.common",
    "build_target_pose": ".pose",
    "evaluated_semantic_bone": ".solve.common",
    "facs_values": ".solve.face",
    "iter_apply_frame": ".pose",
    "iter_apply_sequence": ".sequence",
    "keyframe_interpolation_defaults": ".keyframes",
    "limb_ownership": ".solve.ik",
    "linear_keyframe_defaults": ".keyframes",
    "prepare_limb_ownership": ".solve.ik",
    "source_root_basis": ".solve.frames",
    "target_basis": ".solve.frames",
    "target_rest_basis": ".solve.frames",
    "validate_limb_switch_interval": ".solve.ik",
}
_SUBMODULES = {
    "camera_motion",
    "capture",
    "detection_filter",
    "diagnostics",
    "grounding",
    "keyframes",
    "pose",
    "rohm",
    "root_motion",
    "samples",
    "sequence",
    "solve",
    "target_pose",
    "temporal",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
