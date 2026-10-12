"""Face Expression Cleanup (video only).

Runs before retargeting, on the detected face coefficients, so the sequence
run applies it itself (see ``retarget/sequence.py``); neither the per-frame
nor the sequence pipeline calls this module.
"""

from __future__ import annotations

from ..base import CorrectionModule


class FaceCorrection(CorrectionModule):
    module_id = "FACE_CORRECTION"
    display_name = "Face Expression Cleanup"
    scope = "FACE"
