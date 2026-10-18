from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from media_to_pose.core import face_ranges


def _entry(**changes) -> SimpleNamespace:
    values = {
        "input_zero": 0.0,
        "input_full": 1.0,
        "output_zero": 0.0,
        "output_full": 1.0,
        "output_negative": -1.0,
        "influence": 1.0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _old_value(detected, *, gain, offset, clamp_min, clamp_max, scale):
    """The 0.2.0 face value: clamp((detected - offset) * gain) * travel."""
    return min(max((detected - offset) * gain, clamp_min), clamp_max) * scale


class FaceRangeTests(unittest.TestCase):
    def test_defaults_pass_detections_straight_through(self) -> None:
        entry = _entry()
        for value in (0.0, 0.3, 1.0):
            self.assertAlmostEqual(face_ranges.map_value(entry, value), value)
        self.assertEqual(face_ranges.map_value(entry, 1.4), 1.0)
        self.assertEqual(face_ranges.map_value(entry, -0.5), 0.0)

    def test_input_range_makes_a_weak_blink_close_fully(self) -> None:
        entry = _entry(input_zero=0.05, input_full=0.8)
        self.assertEqual(face_ranges.map_value(entry, 0.8), 1.0)
        self.assertEqual(face_ranges.map_value(entry, 0.04), 0.0)
        self.assertAlmostEqual(face_ranges.map_value(entry, 0.425), 0.5)

    def test_output_range_in_target_units_and_influence(self) -> None:
        lid = _entry(output_full=-0.021, influence=0.5)
        self.assertAlmostEqual(face_ranges.map_value(lid, 1.0), -0.0105)
        key = _entry(output_zero=0.1, output_full=1.3)
        self.assertAlmostEqual(face_ranges.map_value(key, 0.5), 0.7)

    def test_signed_controls_use_the_negative_output(self) -> None:
        brow = _entry(output_full=0.01, output_negative=-0.004)
        self.assertAlmostEqual(face_ranges.map_value(brow, -1.0, signed=True), -0.004)
        self.assertAlmostEqual(face_ranges.map_value(brow, -0.5, signed=True), -0.002)
        self.assertEqual(face_ranges.map_value(brow, -0.5), 0.0)

    def test_share_of_output_inverts_the_mapping(self) -> None:
        entry = _entry(
            output_zero=0.02, output_full=0.06, output_negative=0.0, influence=0.8
        )
        for share in (-1.0, -0.25, 0.0, 0.4, 1.0):
            value = face_ranges.output_value(entry, share)
            self.assertAlmostEqual(face_ranges.share_of_output(entry, value), share)

    def test_legacy_conversion_matches_the_old_formula(self) -> None:
        generator = random.Random(4)
        for _case in range(200):
            gain = generator.choice((0.5, 1.0, 1.5, 3.0, -1.0))
            offset = generator.uniform(-0.2, 0.3)
            clamp_max = generator.choice((1.0, 0.7))
            signed = generator.random() < 0.5
            clamp_min = -clamp_max if signed else 0.0
            scale = generator.choice((1.0, 0.02, 18.0))
            entry = _entry(
                **face_ranges.legacy_ranges(
                    gain=gain, offset=offset, clamp_max=clamp_max, scale=scale
                )
            )
            for detected in (generator.uniform(-1.0, 1.0) for _ in range(10)):
                old = _old_value(
                    detected,
                    gain=gain,
                    offset=offset,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                    scale=scale,
                )
                new = face_ranges.map_value(entry, detected, signed=signed)
                self.assertAlmostEqual(old, new, places=9, msg=(gain, offset, detected))

    def test_legacy_invert(self) -> None:
        entry = _entry(**face_ranges.legacy_ranges(invert=True))
        self.assertAlmostEqual(face_ranges.map_value(entry, 0.25), 0.75)

    def test_legacy_zero_gain_holds_the_target(self) -> None:
        for invert, held in ((False, 0.0), (True, 0.6)):
            entry = _entry(
                **face_ranges.legacy_ranges(
                    gain=0.0, offset=0.2, clamp_max=1.0, invert=invert, scale=0.6
                )
            )
            for detected in (-1.0, 0.0, 0.5, 1.0):
                for signed in (False, True):
                    self.assertAlmostEqual(
                        face_ranges.map_value(entry, detected, signed=signed), held
                    )


if __name__ == "__main__":
    unittest.main()
