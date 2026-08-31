"""Run the CI test suite once inside Blender."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.blender_test_support import unregister_addon  # noqa: E402

TEST_MODULES = (
    "tests.test_core_cache",
    "tests.test_camera_calibration",
    "tests.test_mapping",
    "tests.test_postprocess",
    "tests.test_camera_motion_correction",
    "tests.test_rohm_backend",
    "tests.test_rohm_ort_runtime",
    "tests.test_rohm_correction",
    "tests.test_grounded_cleanup",
    "tests.test_e2e_vroid",
    "tests.test_capture_operations",
    "tests.test_registration",
    "tests.test_anycalib_backend",
    "tests.test_sam3d_backend",
    "tests.test_sam3d_split_bundle",
    "tests.test_sam3d_mhr_runtime",
    "tests.test_sam3d_vith_runtime",
)


def main() -> None:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module_name in TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(module_name))
    try:
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    finally:
        unregister_addon()
    if not result.wasSuccessful():
        raise RuntimeError("Media to Pose test suite failed")


if __name__ == "__main__":
    main()
