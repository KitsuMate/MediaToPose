"""Scripting API: capture, inspect, configure, evaluate and render without UI.

Works inside Blender, in the foreground or with ``blender -b``::

    from media_to_pose import api   # inside Blender: bl_ext.<repo>.media_to_pose

    result = api.capture("clip.mp4", "Rig", parts={"body", "hands"})
    metrics = api.evaluate("Rig", result=result)
    api.retarget_cached(
        "clip.mp4", "Rig", config={"TEMPORAL_CORRECTION": {"smoothing_strength": 0.8}}
    )

Every function raises ``MtpError`` subclasses with a stable ``code``; see
``docs/agents.md`` for recipes and what each metric means.
"""

from __future__ import annotations

from ..core.errors import (
    BackendError,
    Cancelled,
    CaptureFailed,
    InvalidRequest,
    MappingIncomplete,
    MediaError,
    ModelMissing,
    MtpError,
    OperationBusy,
)
from ..evaluate.metrics import Metrics
from ..session.results import CaptureResult
from .capture import capture, retarget_cached
from .config import get_config, list_postprocessors, set_config
from .detections import export_detections, import_detections
from .evaluation import evaluate, load_result, render_comparison
from .models import list_providers, model_status
from .rig import auto_map, describe_rig, fit_face_controls

__all__ = [
    "BackendError",
    "Cancelled",
    "CaptureFailed",
    "CaptureResult",
    "InvalidRequest",
    "MappingIncomplete",
    "MediaError",
    "Metrics",
    "ModelMissing",
    "MtpError",
    "OperationBusy",
    "auto_map",
    "capture",
    "describe_rig",
    "evaluate",
    "export_detections",
    "fit_face_controls",
    "get_config",
    "import_detections",
    "list_postprocessors",
    "list_providers",
    "load_result",
    "model_status",
    "render_comparison",
    "retarget_cached",
    "set_config",
]
