from __future__ import annotations

import importlib
import tempfile
import unittest

import numpy as np

from tests.blender_test_support import PACKAGE_NAME, load_addon

load_addon()
runtime = importlib.import_module(f"{PACKAGE_NAME}.sam3d_mhr_runtime")
MhrCameraOutput = runtime.MhrCameraOutput
MhrModel = runtime.MhrModel
MhrOutput = runtime.MhrOutput
MhrRuntimeError = runtime.MhrRuntimeError
quaternion_xyzw_to_matrix = runtime.quaternion_xyzw_to_matrix


class MhrMathTests(unittest.TestCase):
    def test_quaternion_xyzw_to_matrix(self):
        half = np.float32(np.pi / 4.0)
        quaternions = np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, np.sin(half), np.cos(half)]],
            dtype=np.float32,
        )
        expected = np.array(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            quaternion_xyzw_to_matrix(quaternions), expected, atol=2e-7
        )

    def test_quaternion_rejects_zero_norm(self):
        with self.assertRaisesRegex(MhrRuntimeError, "zero norm"):
            quaternion_xyzw_to_matrix(np.zeros((1, 4), dtype=np.float32))

    def test_camera_space_has_explicit_units_and_axis_conversion(self):
        raw = np.array([[[100.0, 200.0, -300.0]]], dtype=np.float32)
        rotations = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3)
        output = MhrOutput(
            vertices=raw,
            skeleton=np.zeros((1, 1, 8), dtype=np.float32),
            keypoints=raw,
            joint_coords=raw,
            joint_quaternions=np.array([[[0, 0, 0, 1]]], dtype=np.float32),
            joint_scales=np.ones((1, 1), dtype=np.float32),
            joint_rotations=rotations,
        )
        camera = output.camera_space()
        self.assertIsInstance(camera, MhrCameraOutput)
        np.testing.assert_array_equal(
            camera.vertices, np.array([[[1.0, -2.0, 3.0]]], dtype=np.float32)
        )
        self.assertIs(camera.joint_rotations, rotations)

    def test_missing_artifacts_fail_closed(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(MhrRuntimeError, "Missing MHR LBS"),
        ):
            MhrModel.from_directory(directory)


if __name__ == "__main__":
    unittest.main()
