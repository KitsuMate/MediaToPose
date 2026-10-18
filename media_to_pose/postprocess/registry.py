"""The registered clean-up modules, in pipeline order."""

from __future__ import annotations

from .modules.camera_motion import CameraMotionCorrection
from .modules.detection_filter import DetectionFilter
from .modules.limb_axis import ArmAxisCorrection, LegAxisCorrection
from .modules.rohm import RohmCorrection
from .modules.shoulders import ShoulderCorrection
from .modules.spine import SpineCorrection
from .modules.temporal import TemporalCorrection

CORRECTION_REGISTRY = {
    module.module_id: module
    for module in (
        SpineCorrection(),
        ShoulderCorrection(),
        ArmAxisCorrection(),
        LegAxisCorrection(),
        CameraMotionCorrection(),
        DetectionFilter(),
        RohmCorrection(),
        TemporalCorrection(),
    )
}
