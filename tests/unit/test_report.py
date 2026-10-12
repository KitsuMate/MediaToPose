from __future__ import annotations

import unittest

from media_to_pose.core import report


class CaptureReportTests(unittest.TestCase):
    def test_records_only_while_active(self) -> None:
        collected = report.CaptureReport(source_frame_offset=10)
        report.record_step("DETECTION_FILTER", rejected_frames=1)
        self.assertEqual(collected.steps, {})
        with collected.active():
            report.record_step("DETECTION_FILTER", {"maximum": 5}, rejected_frames=2)
            report.record_step("DETECTION_FILTER", repaired_samples=3)
            report.record_frames("rejected", [0, 1, 4])
        self.assertIsNone(report.active_report())
        step = collected.steps["DETECTION_FILTER"].to_dict()
        self.assertEqual(step["parameters"], {"maximum": 5})
        self.assertEqual(
            step["statistics"], {"rejected_frames": 2, "repaired_samples": 3}
        )
        # Sample indices become source frames of the processed range.
        self.assertEqual(collected.rejected_frames, {10, 11, 14})

    def test_frame_ranges_collapse_runs(self) -> None:
        self.assertEqual(
            report.frame_ranges([5, 1, 2, 3, 9, 6]), [(1, 3), (5, 6), (9, 9)]
        )
        self.assertEqual(report.frame_ranges([]), [])

    def test_statistics_become_json_values(self) -> None:
        step = report.StepReport(
            "X", statistics={"ratio": 1 / 3, "frames": {3, 1}, "pair": (1, 2)}
        )
        self.assertEqual(
            step.to_dict()["statistics"],
            {"ratio": 0.333333, "frames": [1, 3], "pair": [1, 2]},
        )


if __name__ == "__main__":
    unittest.main()
