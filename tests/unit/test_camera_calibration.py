from __future__ import annotations

import math
import unittest

import numpy as np

from media_to_pose.core.camera import calibration as camera_calibration


class CameraCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = camera_calibration

    def test_fallback_retains_diagonal_focal_contract(self):
        value = self.calibration.fallback(1920, 1080)
        self.assertEqual(value.source, "FALLBACK")
        self.assertAlmostEqual(value.fx, math.hypot(1920, 1080))
        self.assertAlmostEqual(value.cx, 960.0)

    def test_manual_vertical_fov_uses_square_pixels(self):
        value = self.calibration.manual(1920, 1080, 40.0)
        self.assertEqual(value.source, "MANUAL")
        self.assertAlmostEqual(value.vertical_fov, 40.0, places=6)
        self.assertAlmostEqual(value.fx, value.fy)

    def test_flip_mirrors_only_principal_x(self):
        value = self.calibration.intrinsics(
            100, 80, 120.0, 121.0, 43.0, 37.0, source="ANYCALIB"
        )
        flipped = self.calibration.mirrored(value)
        self.assertAlmostEqual(flipped.cx, 57.0)
        self.assertAlmostEqual(flipped.cy, 37.0)
        self.assertAlmostEqual(flipped.fx, 120.0)

    def test_dense_tangent_fit_recovers_intrinsics(self):
        height, width = 56, 84
        fx, fy, cx, cy = 100.0, 95.0, 39.0, 25.0
        x, y = np.meshgrid(
            np.arange(width, dtype=float) + 0.5,
            np.arange(height, dtype=float) + 0.5,
            indexing="xy",
        )
        projected = np.stack(((x - cx) / fx, (y - cy) / fy), axis=-1)
        radii = np.linalg.norm(projected, axis=-1, keepdims=True)
        angles = np.arctan(radii)
        tangent = projected * np.where(radii > 1.0e-9, angles / radii, 1.0)
        field = tangent.transpose(2, 0, 1)[None]
        value = self.calibration.fit_tangent_field(
            field,
            original_width=width,
            original_height=height,
            np=np,
        )
        self.assertAlmostEqual(value.fx, fx, places=5)
        self.assertAlmostEqual(value.fy, fy, places=5)
        self.assertAlmostEqual(value.cx, cx, places=5)
        self.assertAlmostEqual(value.cy, cy, places=5)

    def test_video_consensus_rejects_zoom_range(self):
        values = [
            self.calibration.manual(1000, 1000, fov) for fov in (30.0, 31.0, 50.0)
        ]
        for value in values:
            value.source = "ANYCALIB"
        self.assertIsNone(
            self.calibration.consensus(values, minimum=3, maximum_fov_range=8.0)
        )

    def test_image_consensus_retains_primary_when_augmented_views_disagree(self):
        values = [
            self.calibration.manual(1000, 1000, fov) for fov in (44.1, 45.8, 48.7)
        ]
        for value in values:
            value.source = "ANYCALIB"
        result = self.calibration.image_consensus(values)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.vertical_fov, 44.1, places=6)

    def test_even_consensus_uses_mathematical_median(self):
        values = [self.calibration.manual(1000, 1000, fov) for fov in (40.0, 42.0)]
        for value in values:
            value.source = "ANYCALIB"
        result = self.calibration.consensus(values, minimum=2)
        expected_focal = 0.5 * (values[0].fy + values[1].fy)
        self.assertAlmostEqual(result.fy, expected_focal, places=6)


if __name__ == "__main__":
    unittest.main()
