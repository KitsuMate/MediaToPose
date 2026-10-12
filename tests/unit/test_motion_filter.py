"""Confidence and temporal filtering before motion reconstruction."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from media_to_pose.core.geometry import Quaternion, Vector
from media_to_pose.core.motion import motion_filter
from media_to_pose.core.motion.rohm_correction import SMPLX_SEMANTICS


def _channel(semantic: str, degrees: float, confidence: float = 1.0):
    return SimpleNamespace(
        semantic_id=semantic,
        key_rotation=True,
        key_location=False,
        rotation=Quaternion((0.0, 0.0, 1.0), math.radians(degrees)),
        location=Vector((0.0, 0.0, 0.0)),
        scale=Vector((1.0, 1.0, 1.0)),
        properties={},
        confidence=confidence,
    )


def _sample(degrees: float, confidence: float = 1.0):
    return SimpleNamespace(
        bones={
            semantic: _channel(semantic, degrees, confidence)
            for semantic in SMPLX_SEMANTICS
        },
        rohm_confidence={semantic: confidence for semantic in SMPLX_SEMANTICS},
        rohm_rotations={
            semantic: Quaternion((0.0, 0.0, 1.0), math.radians(degrees))
            for semantic in SMPLX_SEMANTICS
        },
        rohm_root_confidence=confidence,
        source_body={},
        grounding_body={},
    )


class SourceFrameRangeTests(unittest.TestCase):
    def test_parses_one_based_frames_and_ranges(self) -> None:
        self.assertEqual(
            motion_filter.parse_source_frame_ranges("1, 3-5", 10),
            frozenset({0, 2, 3, 4}),
        )

    def test_clips_to_the_frame_count(self) -> None:
        self.assertEqual(
            motion_filter.parse_source_frame_ranges("8-20", 10),
            frozenset({7, 8, 9}),
        )

    def test_empty_text_selects_nothing(self) -> None:
        self.assertEqual(motion_filter.parse_source_frame_ranges("  ", 10), set())

    def test_rejects_malformed_ranges(self) -> None:
        for text in ("0", "5-3", "a", "1,,2", "1-2-3", "-4"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                motion_filter.parse_source_frame_ranges(text, 10)

    def test_formats_compact_ranges(self) -> None:
        self.assertEqual(
            motion_filter.format_source_frame_ranges([0, 1, 2, 5, 7, 8]),
            "1-3, 6, 8-9",
        )
        self.assertEqual(motion_filter.format_source_frame_ranges([]), "None")

    def test_format_and_parse_round_trip(self) -> None:
        indices = frozenset({0, 1, 4, 9, 10, 11})
        text = motion_filter.format_source_frame_ranges(indices)
        self.assertEqual(motion_filter.parse_source_frame_ranges(text, 20), indices)


class ManualRejectionTests(unittest.TestCase):
    def test_arm_scope_masks_only_that_limb(self) -> None:
        samples = [_sample(0.0) for _ in range(4)]
        rejected = motion_filter.apply_manual_motion_rejections(
            samples, [1, 2, 99], "arm_l"
        )
        self.assertEqual(
            rejected,
            frozenset(
                (frame, semantic)
                for frame in (1, 2)
                for semantic in motion_filter.MANUAL_MASK_SCOPES["ARM_L"]
            ),
        )
        self.assertEqual(samples[1].rohm_confidence["HAND_L"], 0.0)
        self.assertEqual(samples[1].bones["HAND_L"].confidence, 0.0)
        self.assertEqual(samples[1].rohm_confidence["HAND_R"], 1.0)
        self.assertEqual(samples[1].rohm_root_confidence, 1.0)

    def test_body_scope_also_masks_the_root(self) -> None:
        samples = [_sample(0.0) for _ in range(2)]
        motion_filter.apply_manual_motion_rejections(samples, [0], "BODY")
        self.assertEqual(samples[0].rohm_root_confidence, 0.0)

    def test_unknown_scope_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            motion_filter.apply_manual_motion_rejections([], [], "TAIL")


class AutomaticFilterTests(unittest.TestCase):
    def test_weak_limb_is_rejected_within_budget(self) -> None:
        samples = [_sample(float(index)) for index in range(6)]
        for semantic in motion_filter.MANUAL_MASK_SCOPES["LEG_R"]:
            samples[3].rohm_confidence[semantic] = 0.1
        stats = motion_filter.filter_motion_observations(
            samples,
            [index / 30.0 for index in range(6)],
            reject_unreliable_limbs=True,
            reject_unusable_frames=False,
            maximum_rejected_fraction=0.5,
        )
        self.assertEqual(stats.rejected_limbs, 1)
        self.assertEqual(
            {semantic for _frame, semantic in stats.rejected},
            set(motion_filter.MANUAL_MASK_SCOPES["LEG_R"]),
        )
        self.assertEqual(samples[3].rohm_confidence["FOOT_R"], 0.0)

    def test_zero_budget_rejects_nothing(self) -> None:
        samples = [_sample(float(index)) for index in range(6)]
        for semantic in SMPLX_SEMANTICS:
            samples[2].rohm_confidence[semantic] = 0.05
        stats = motion_filter.filter_motion_observations(
            samples,
            [index / 30.0 for index in range(6)],
            reject_unreliable_limbs=True,
            reject_unusable_frames=True,
            maximum_rejected_fraction=0.0,
        )
        self.assertEqual(stats.rejection_budget, 0)
        self.assertEqual(stats.rejected_observations, 0)

    def test_mismatched_timestamps_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            motion_filter.filter_motion_observations(
                [_sample(0.0)],
                [],
                reject_unreliable_limbs=True,
                reject_unusable_frames=True,
                maximum_rejected_fraction=0.1,
            )


class GapRepairTests(unittest.TestCase):
    def test_internal_gap_is_interpolated_in_time(self) -> None:
        samples = [_sample(degrees) for degrees in (0.0, 10.0, 20.0, 30.0)]
        timestamps = [0.0, 0.1, 0.2, 0.3]
        samples[1].rohm_confidence["HEAD"] = 0.0
        samples[2].rohm_confidence["HEAD"] = 0.0
        samples[1].bones["HEAD"].rotation = Quaternion((1.0, 0.0, 0.0), 2.0)
        repaired = motion_filter.repair_body_gaps(samples, timestamps)
        self.assertEqual(repaired, 2)
        self.assertAlmostEqual(
            math.degrees(samples[1].bones["HEAD"].rotation.angle), 10.0, places=4
        )
        self.assertAlmostEqual(
            math.degrees(samples[2].rohm_rotations["HEAD"].angle), 20.0, places=4
        )

    def test_boundary_gap_holds_the_nearest_good_pose(self) -> None:
        samples = [_sample(degrees) for degrees in (5.0, 15.0, 25.0)]
        samples[0].rohm_confidence["NECK"] = 0.0
        motion_filter.repair_body_gaps(samples, [0.0, 0.1, 0.2])
        self.assertAlmostEqual(
            math.degrees(samples[0].bones["NECK"].rotation.angle), 15.0, places=4
        )


if __name__ == "__main__":
    unittest.main()
