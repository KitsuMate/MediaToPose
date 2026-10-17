from __future__ import annotations

import math
import unittest

from media_to_pose.core.motion import face_filter

FPS = 30.0


def _times(count: int) -> list[float]:
    return [index / FPS for index in range(count)]


def _settings(**changes) -> face_filter.FaceFilterSettings:
    values = {
        "smooth": False,
        "preserve_blinks": False,
        "neutral_calibration": False,
        "deadzone": 0.0,
        "hold_seconds": 0.0,
    }
    values.update(changes)
    return face_filter.FaceFilterSettings(**values)


class FaceFilterTests(unittest.TestCase):
    def test_missing_face_is_held_then_faded(self) -> None:
        track = [{"jawOpen": 0.8}] * 3 + [None] * 30
        result, stats = face_filter.filter_face_track(
            track, _times(len(track)), _settings(hold_seconds=0.3)
        )
        self.assertAlmostEqual(result[5]["jawOpen"], 0.8)
        # Held for 0.3 s (9 frames), faded over the next 0.3 s, then gone.
        self.assertLess(result[16]["jawOpen"], 0.8)
        self.assertEqual(result[-1], {"jawOpen": 0.0})
        self.assertEqual(stats.frames_without_face, 30)
        self.assertEqual(stats.held_frames, 9)

    def test_leading_missing_frames_stay_empty(self) -> None:
        track = [None, None, {"jawOpen": 0.8}, None]
        result, _ = face_filter.filter_face_track(track, _times(4), _settings())
        self.assertEqual(result[:2], [{}, {}])
        self.assertEqual(result[-1], {"jawOpen": 0.0})

    def test_neutral_calibration_removes_a_resting_bias(self) -> None:
        # Arched brows rest at 0.8 and rise to 1.0 for a moment.
        values = [0.8] * 40 + [1.0] * 5 + [0.8] * 15
        track = [{"browOuterUpLeft": v, "jawOpen": 0.0} for v in values]
        result, stats = face_filter.filter_face_track(
            track, _times(len(track)), _settings(neutral_calibration=True)
        )
        self.assertAlmostEqual(result[0]["browOuterUpLeft"], 0.0, places=6)
        self.assertAlmostEqual(result[42]["browOuterUpLeft"], 1.0, places=6)
        self.assertEqual(stats.calibrated, {"browOuterUpLeft": 0.8})

    def test_deadzone_silences_detector_noise(self) -> None:
        track = [{"mouthSmileLeft": 0.03}, {"mouthSmileLeft": 0.55}]
        result, _stats = face_filter.filter_face_track(
            track, _times(2), _settings(deadzone=0.05)
        )
        self.assertEqual(result[0]["mouthSmileLeft"], 0.0)
        self.assertAlmostEqual(result[1]["mouthSmileLeft"], 0.5 / 0.95)

    def test_blinks_survive_smoothing(self) -> None:
        values = [0.05] * 20 + [0.6, 0.95, 0.9, 0.5] + [0.05] * 20
        track = [{"eyeBlinkLeft": v} for v in values]
        plain, _ = face_filter.filter_face_track(
            track, _times(len(track)), _settings(smooth=True, strength=1.0)
        )
        kept, stats = face_filter.filter_face_track(
            track,
            _times(len(track)),
            _settings(smooth=True, strength=1.0, preserve_blinks=True),
        )
        self.assertLess(plain[21]["eyeBlinkLeft"], 0.8)
        self.assertAlmostEqual(kept[21]["eyeBlinkLeft"], 0.95)
        self.assertEqual(stats.blinks, 1)
        # A long closed-eye stretch is not a blink.
        long_close = [0.9] * 30
        self.assertEqual(face_filter.find_blinks(long_close, _times(30), 0.5, 0.5), [])

    def test_smoothing_reduces_jitter_without_lag(self) -> None:
        values = [0.5 + 0.1 * math.sin(index * 2.5) for index in range(60)]
        track = [{"mouthSmileLeft": v} for v in values]
        result, _ = face_filter.filter_face_track(
            track, _times(60), _settings(smooth=True, strength=1.0)
        )
        smoothed = [frame["mouthSmileLeft"] for frame in result]

        def roughness(series):
            return sum(abs(series[i + 1] - series[i]) for i in range(len(series) - 1))

        self.assertLess(roughness(smoothed), roughness(values) * 0.5)
        self.assertAlmostEqual(sum(smoothed) / 60, sum(values) / 60, delta=0.02)

    def test_groups(self) -> None:
        group = face_filter.coefficient_group
        self.assertEqual(group("eyeBlinkLeft"), "eye")
        self.assertEqual(group("browInnerUp"), "brow")
        self.assertEqual(group("jawOpen"), "jaw")
        self.assertEqual(group("mouthPucker"), "mouth")


if __name__ == "__main__":
    unittest.main()
