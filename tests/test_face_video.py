"""Face video sequences through Smooth Motion's face cleanup, and face metrics.

The blink, calibration and smoothing maths are unit-tested in
``tests/unit/test_face_filter.py``; these tests check the sequence wiring
and what the keyed rig shows.
"""

from __future__ import annotations

import math
import unittest

import bpy

from tests.blender_test_support import (
    face_settings,
    open_rig,
    register_addon,
    scene_context,
)


def _blink_clip(count: int = 40) -> list[dict[str, float]]:
    """A clip with a resting smile bias, jitter, and one quick blink."""
    clip = []
    for index in range(count):
        blink = {20: 0.7, 21: 0.95, 22: 0.9, 23: 0.6}.get(index, 0.05)
        clip.append(
            {
                "eyeBlinkLeft": blink,
                "eyeBlinkRight": blink,
                "mouthSmileLeft": 0.3 + 0.12 * math.sin(index * 2.5),
            }
        )
    return clip


def _roughness(values: list[float]) -> float:
    return sum(abs(values[i + 1] - values[i]) for i in range(len(values) - 1))


class FaceVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.rig = open_rig(
            cls.addon,
            "facefx_face_rig",
            "FaceFX.Rig",
            use_face=True,
            use_body=False,
            use_hands=False,
        )
        cls.keys = bpy.data.objects["FaceFX"].data.shape_keys

    def _run(
        self,
        clip: list[dict[str, float]],
        *,
        cleanup: bool = True,
        neutral_calibration: bool = True,
        face_output: str = "ARKIT",
    ) -> tuple[dict[str, list[float]], list]:
        """Key ``clip``; the keyed blink and smile values, and the detections."""
        config = self.rig.mtp_rig_config
        saved = [(item, item.enabled) for item in config.post_processing_modules]
        self.addCleanup(lambda: [setattr(i, "enabled", v) for i, v in saved])
        for item in config.post_processing_modules:
            item.enabled = False
        # Smooth Motion cleans the face; without it the raw values are keyed.
        temporal = self.addon.postprocess.module_config(config, "TEMPORAL_CORRECTION")
        temporal.enabled = cleanup
        self.addCleanup(setattr, temporal, "neutral_calibration", True)
        temporal.neutral_calibration = neutral_calibration

        context = scene_context()
        step_ms = 1000.0 / context.fps
        frames = [
            self.addon.core.types.CanonicalFrame(
                timestamp_ms=round(index * step_ms), blendshapes=values
            )
            for index, values in enumerate(clip)
        ]
        for animated in (self.keys, self.rig):
            animated.animation_data_clear()
            self.addCleanup(animated.animation_data_clear)
        self.addCleanup(
            lambda: [b.matrix_basis.identity() for b in self.rig.pose.bones]
        )
        settings = face_settings(
            face_output,
            media_kind="VIDEO",
            root_motion_mode="OFF",
            keep_original_root_position=False,
            keep_detection_cache=False,
        )
        self.addon.retarget.apply_sequence(
            self.rig, frames, settings, 1.0, scene_context=context
        )
        # The face carries arkit_ and facs_bs_ copies; arkit_ is keyed.
        names = {"eyeBlinkLeft": "arkit_EyeBlinkLeft", "smile": "arkit_MouthSmileLeft"}
        values: dict[str, list[float]] = {role: [] for role in names}
        for index in range(len(frames)):
            bpy.context.scene.frame_set(1 + index)
            for role, name in names.items():
                values[role].append(self.keys.key_blocks[name].value)
        detections = [(1.0 + index, frame) for index, frame in enumerate(frames)]
        return values, detections

    def _metrics(self, detections, face_output: str = "ARKIT"):
        return self.addon.evaluate.metrics.evaluate(
            scene_context(),
            self.rig,
            frame_range=(1, len(detections)),
            detections=detections,
            face_output=face_output,
        )

    def test_cleanup_keeps_blinks_calibrates_and_smooths(self) -> None:
        raw, detections = self._run(_blink_clip(), cleanup=False)
        raw_metrics = self._metrics(detections)
        cleaned, _detections = self._run(_blink_clip())
        metrics = self._metrics(detections)
        # The blink keeps nearly its full depth at the detected frame.
        self.assertGreater(cleaned["eyeBlinkLeft"][21], 0.85)
        self.assertLess(max(cleaned["eyeBlinkLeft"][:15]), 0.05)
        self.assertEqual(metrics.blink_recall, 1.0)
        # The jittery smile is smoothed and its resting level removed.
        smile = cleaned["smile"]
        self.assertLess(_roughness(smile), 0.5 * _roughness(raw["smile"]))
        self.assertLess(sum(smile) / len(smile), 0.25)
        self.assertLess(metrics.face_jitter, raw_metrics.face_jitter)
        # Cleaned values differ from the detections on purpose.
        self.assertGreater(metrics.expression_fit, raw_metrics.expression_fit)
        self.assertLess(metrics.asymmetry, 0.05)

    def test_without_neutral_calibration_the_resting_smile_stays(self) -> None:
        values, _detections = self._run(_blink_clip(), neutral_calibration=False)
        self.assertGreater(sum(values["smile"]) / len(values["smile"]), 0.2)

    def test_lost_face_keys_neutral_after_fading(self) -> None:
        # Fewer than ten detections avoids neutral calibration; the long gap
        # must end in explicit zero keys instead of holding the last smile.
        clip = [{"mouthSmileLeft": 0.8, "eyeBlinkLeft": 0.0}] * 3 + [{}] * 60
        values, _detections = self._run(clip)
        self.assertGreater(values["smile"][0], 0.5)
        self.assertAlmostEqual(values["smile"][-1], 0.0, places=6)

    def test_face_metrics_on_bones(self) -> None:
        _values, detections = self._run(_blink_clip(), face_output="BONES")
        measured = self._metrics(detections, "BONES")
        self.assertEqual(measured.blink_recall, 1.0)
        self.assertLess(measured.asymmetry, 0.05)
        self.assertIsNotNone(measured.expression_fit)
        # Without a face output there are no face metrics.
        self.assertIsNone(
            self.addon.evaluate.metrics.evaluate(
                scene_context(), self.rig, frame_range=(1, 5)
            ).face_jitter
        )


if __name__ == "__main__":
    unittest.main()
