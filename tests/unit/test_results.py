from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from media_to_pose.core.report import CaptureReport, record_step
from media_to_pose.core.requests import PostProcessConfig, PostProcessModuleConfig
from media_to_pose.core.types import CanonicalFrame, Landmark
from media_to_pose.session import results


class CaptureResultTests(unittest.TestCase):
    def test_warning_codes_are_stable(self) -> None:
        self.assertEqual(
            results.warning_code("Detection Filter rejected 4 sample(s)"),
            "DETECTIONS_REJECTED",
        )
        self.assertEqual(
            results.warning_code("Ground Contacts repaired 2 isolated foot sample(s)"),
            "GROUND_CONTACTS",
        )
        self.assertEqual(results.warning_code("Something new"), "CAPTURE_WARNING")

    def test_part_confidence(self) -> None:
        frame = CanonicalFrame(
            timestamp_ms=0,
            body={
                "NOSE": Landmark((0.0, 0.0, 0.0), confidence=0.8),
                "LEFT_HIP": Landmark((0.0, 0.0, 0.0), confidence=0.4),
                "RIGHT_HIP": Landmark((0.0, 0.0, 0.0), confidence=0.9, valid=False),
            },
            blendshapes={"jawOpen": 0.2},
        )
        self.assertEqual(
            results.part_confidence(frame), {"body": 0.6, "hands": 0.0, "face": 1.0}
        )

    def test_cleanup_steps_merge_settings_and_statistics(self) -> None:
        config = PostProcessConfig(
            detected_profile="GENERIC",
            modules=(
                PostProcessModuleConfig(
                    "TEMPORAL_CORRECTION", "Temporal", True, {"smoothing_strength": 0.4}
                ),
                PostProcessModuleConfig("ROHM_CORRECTION", "RoHM", False),
            ),
        )
        request = type("Request", (), {"postprocess_config": config})()
        report = CaptureReport()
        with report.active():
            record_step("TEMPORAL_CORRECTION", contact_frames=12)
        steps = results.cleanup_steps(request, report)
        self.assertEqual(
            steps,
            [
                {
                    "module_id": "TEMPORAL_CORRECTION",
                    "display_name": "Temporal",
                    "parameters": {"smoothing_strength": 0.4},
                    "statistics": {"contact_frames": 12},
                }
            ],
        )

    def test_result_round_trips_through_json(self) -> None:
        result = results.CaptureResult(
            status="SUCCEEDED",
            message="done",
            mode="KEYFRAMES",
            provider="MEDIAPIPE",
            media={"path": "clip.mp4", "kind": "VIDEO"},
            armature="Rig",
            rejected_frames=[[3, 5]],
            warnings=[{"code": "GAPS_FILLED", "message": "filled"}],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = result.write_json(Path(directory) / "result.json")
            stored = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(stored, result.to_dict())
        self.assertEqual(stored["schema_version"], results.RESULT_SCHEMA_VERSION)
        self.assertTrue(result.succeeded)


if __name__ == "__main__":
    unittest.main()
