"""Time-aware animation cleanup helpers."""

from __future__ import annotations

import math
import unittest

from media_to_pose import animation_cleanup
from media_to_pose.geometry import Quaternion, Vector


class AnimationCleanupTests(unittest.TestCase):
    def test_isolated_scalar_and_vector_spikes_are_rejected(self) -> None:
        noisy = [0.0, 1.0, 2.0, 20.0, 4.0, 5.0, 6.0]
        times = [float(index) for index in range(len(noisy))]
        cleaned, _count = animation_cleanup.reduce_scalar_outliers(noisy, times)
        cleaned = animation_cleanup.smooth_scalars(cleaned, times, None, 1.0)
        self.assertLessEqual(abs(cleaned[3] - 3.0), abs(noisy[3] - 3.0) * 0.2)
        self.assertLess(abs(cleaned[0] - noisy[0]), 1.0)
        vectors = [Vector((value, value * 0.5, 0.0)) for value in noisy]
        cleaned_vectors, _count = animation_cleanup.reduce_vector_outliers(
            vectors, times
        )
        cleaned_vectors = animation_cleanup.smooth_vectors(
            cleaned_vectors, times, None, 1.0
        )
        self.assertLessEqual(
            abs(cleaned_vectors[3].x - 3.0),
            abs(vectors[3].x - 3.0) * 0.2,
        )

    def test_quaternion_cleanup_keeps_short_continuous_hemisphere(self) -> None:
        crossing = [
            Quaternion((0.0, 0.0, 1.0), math.radians(degrees))
            for degrees in (170.0, 179.0, -179.0, -170.0)
        ]
        times = [float(index) for index in range(len(crossing))]
        cleaned = animation_cleanup.smooth_quaternions(crossing, times, None, 0.7)
        for first, second in zip(cleaned, cleaned[1:], strict=False):
            self.assertGreaterEqual(first.dot(second), 0.0)
            self.assertLess(
                first.rotation_difference(second).angle,
                math.radians(25.0),
            )

    def test_isolated_half_turn_is_removed(self) -> None:
        samples = [
            Quaternion(),
            Quaternion((0.0, 1.0, 0.0), math.radians(3.0)),
            Quaternion((0.0, 1.0, 0.0), math.radians(180.0)),
            Quaternion((0.0, 1.0, 0.0), math.radians(5.0)),
            Quaternion((0.0, 1.0, 0.0), math.radians(7.0)),
        ]
        times = [float(index) for index in range(len(samples))]
        cleaned, _count = animation_cleanup.reduce_quaternion_outliers(samples, times)
        cleaned = animation_cleanup.smooth_quaternions(cleaned, times, None, 1.0)
        self.assertLess(cleaned[2].angle, math.radians(20.0))

    def test_short_hand_rotation_burst_is_bridged(self) -> None:
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (0.0, 4.0, 82.0, -70.0, 7.0, 10.0)
        ]
        times = [index / 24.0 for index in range(len(values))]
        cleaned, count = animation_cleanup.reduce_short_quaternion_bursts(
            values,
            times,
            minimum_excursion=math.radians(30.0),
        )
        self.assertGreaterEqual(count, 2)
        self.assertLess(
            max(
                math.degrees(left.rotation_difference(right).angle)
                for left, right in zip(cleaned, cleaned[1:], strict=False)
            ),
            20.0,
        )

    def test_sustained_fast_hand_turn_is_preserved(self) -> None:
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (0.0, 25.0, 55.0, 90.0, 125.0, 155.0)
        ]
        times = [index / 24.0 for index in range(len(values))]
        cleaned, count = animation_cleanup.reduce_short_quaternion_bursts(
            values,
            times,
            minimum_excursion=math.radians(30.0),
        )
        self.assertEqual(count, 0)
        self.assertLess(
            cleaned[-1].rotation_difference(values[-1]).angle,
            1.0e-6,
        )

    def test_distal_finger_speed_is_bounded(self) -> None:
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (0.0, 8.0, 130.0, 142.0)
        ]
        times = [index / 24.0 for index in range(len(values))]
        cleaned, count = animation_cleanup.limit_quaternion_speed(
            values,
            times,
            math.radians(1200.0),
        )
        self.assertGreaterEqual(count, 1)
        self.assertLessEqual(
            max(
                math.degrees(left.rotation_difference(right).angle)
                for left, right in zip(cleaned, cleaned[1:], strict=False)
            ),
            50.0001,
        )

    def test_time_aware_smoothing_preserves_linear_motion(self) -> None:
        times = [0.0, 0.03, 0.08, 0.12, 0.20, 0.27]
        values = [2.0 * value + 1.0 for value in times]
        cleaned = animation_cleanup.smooth_scalars(values, times, None, 1.0)
        for expected, actual in zip(values, cleaned, strict=True):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_vector_outlier_uses_vector_residual(self) -> None:
        times = [index / 30.0 for index in range(7)]
        values = [Vector((float(index), 0.0, 0.0)) for index in range(7)]
        values[3] = Vector((3.0, 20.0, 0.0))
        cleaned, count = animation_cleanup.reduce_vector_outliers(values, times)
        self.assertEqual(count, 1)
        self.assertLess(cleaned[3].y, 1.0)

    def test_default_smoothing_halves_stationary_jitter_without_phase_shift(
        self,
    ) -> None:
        times = [index / 30.0 for index in range(61)]
        values = [0.1 if index % 2 else -0.1 for index in range(61)]
        cleaned = animation_cleanup.smooth_scalars(values, times, None, 0.5)
        before = math.sqrt(sum(value * value for value in values) / len(values))
        after = math.sqrt(sum(value * value for value in cleaned) / len(cleaned))
        self.assertLessEqual(after, before * 0.5)

        event = [0.0] * 21
        for offset, value in enumerate((0.2, 0.65, 1.0, 0.65, 0.2)):
            event[8 + offset] = value
        event_times = [index / 30.0 for index in range(len(event))]
        stabilized = animation_cleanup.smooth_scalars(event, event_times, None, 0.5)
        self.assertEqual(stabilized.index(max(stabilized)), 10)
        self.assertGreaterEqual(max(stabilized), 0.9)

    def test_quaternion_smoothing_stays_normalized(self) -> None:
        times = [index / 30.0 for index in range(12)]
        values = [
            Quaternion((0.2, 0.8, 0.1), math.radians(index * 7.0))
            for index in range(12)
        ]
        cleaned = animation_cleanup.smooth_quaternions(values, times, None, 0.7)
        for value in cleaned:
            self.assertAlmostEqual(value.magnitude, 1.0, places=6)

    def test_bounded_quaternion_smoothing_does_not_overshoot_axis_range(self) -> None:
        times = [index / 24.0 for index in range(7)]
        values = [
            Quaternion((0.0, 0.0, 1.0), math.radians(degrees))
            for degrees in (0.0, 5.0, 20.0, 45.0, 70.0, 85.0, 90.0)
        ]

        cleaned = animation_cleanup.smooth_quaternions_bounded(
            values,
            times,
            None,
            1.0,
        )

        for value in cleaned:
            # Pure Z rotation: the Euler Z angle is twice atan2(z, w).
            angle = math.degrees(2.0 * math.atan2(value.z, value.w))
            self.assertGreaterEqual(angle, -1.0e-5)
            self.assertLessEqual(angle, 90.0 + 1.0e-5)
            self.assertAlmostEqual(value.magnitude, 1.0, places=6)

    def test_hand_stabilizer_holds_sustained_detector_burst(self) -> None:
        times = [index / 25.0 for index in range(24)]
        angles = [0.0] * 5 + [82.0] * 13 + [0.0] * 6
        values = [Quaternion((0.0, 0.0, 1.0), math.radians(angle)) for angle in angles]

        cleaned, held = animation_cleanup.stabilize_quaternion_jumps(
            values,
            times,
            math.radians(600.0),
            reacquire_seconds=0.8,
        )

        self.assertEqual(held, 13)
        self.assertLess(
            max(value.rotation_difference(values[0]).angle for value in cleaned),
            math.radians(0.01),
        )

    def test_hand_stabilizer_reacquires_a_persistent_new_pose_boundedly(self) -> None:
        times = [index / 25.0 for index in range(32)]
        values = [Quaternion() for _index in range(3)] + [
            Quaternion((0.0, 0.0, 1.0), math.radians(90.0)) for _index in range(29)
        ]

        cleaned, held = animation_cleanup.stabilize_quaternion_jumps(
            values,
            times,
            math.radians(600.0),
            reacquire_seconds=0.8,
        )

        self.assertGreaterEqual(held, 20)
        self.assertLess(cleaned[20].rotation_difference(values[0]).angle, 1.0e-6)
        self.assertLess(
            cleaned[-1].rotation_difference(values[-1]).angle,
            math.radians(0.01),
        )
        self.assertLessEqual(
            max(
                cleaned[index - 1].rotation_difference(cleaned[index]).angle
                for index in range(1, len(cleaned))
            ),
            math.radians(24.0) + 1.0e-6,
        )

    def test_weak_wrist_interval_holds_then_eases_without_default_pose(self) -> None:
        times = [index * 0.1 for index in range(8)]
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (10.0, 100.0, -80.0, 120.0, -110.0, 90.0, -70.0, 40.0)
        ]
        confidences = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        bridged = animation_cleanup.bridge_low_confidence_quaternions(
            values, times, confidences
        )
        self.assertLess(
            bridged[3].rotation_difference(values[0]).angle,
            math.radians(0.1),
        )
        self.assertGreater(math.degrees(bridged[6].angle), 12.0)
        self.assertLess(math.degrees(bridged[6].angle), 40.0)
        for first, second in zip(bridged, bridged[1:], strict=False):
            self.assertGreaterEqual(first.dot(second), 0.0)

    def test_weak_ik_location_interval_uses_neighboring_motion(self) -> None:
        times = [index / 30.0 for index in range(7)]
        values = [Vector((float(index), 0.0, 0.0)) for index in range(7)]
        values[2].y = 20.0
        values[3].y = -20.0
        values[4].y = 15.0
        confidences = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        bridged = animation_cleanup.bridge_low_confidence_vectors(
            values, times, confidences
        )
        for index in (2, 3, 4):
            self.assertAlmostEqual(bridged[index].y, 0.0, places=6)
