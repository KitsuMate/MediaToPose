"""Run the CI test suite once inside Blender.

Pass ``-- --update-golden`` after the Blender arguments to rewrite the golden
retarget snapshots instead of comparing against them.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.blender_test_support import unregister_addon  # noqa: E402

TEST_MODULES = (
    "tests.test_core_cache",
    "tests.test_mapping",
    "tests.test_postprocess",
    "tests.test_camera_motion_correction",
    "tests.test_rohm_backend",
    "tests.test_e2e_vroid",
    "tests.test_golden_retarget",
    "tests.test_capture_operations",
    "tests.test_capture_session",
    "tests.test_api",
    "tests.test_face_detection",
    "tests.test_face_transfer",
    "tests.test_golden_face",
    "tests.test_registration",
    "tests.test_settings_schema",
    "tests.test_anycalib_backend",
    "tests.test_model_backends",
    "tests.test_sam3d_backend",
    "tests.test_sam3d_split_bundle",
    "tests.test_sam3d_vith_runtime",
)


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if "--update-golden" in arguments:
        os.environ["MTP_UPDATE_GOLDEN"] = "1"
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    # Bare arguments select modules by name, e.g. ``-- test_face_transfer``.
    selected = [name for name in arguments if not name.startswith("--")]
    for module_name in TEST_MODULES:
        if selected and module_name.removeprefix("tests.") not in selected:
            continue
        suite.addTests(loader.loadTestsFromName(module_name))
    try:
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    finally:
        unregister_addon()
    if not result.wasSuccessful():
        raise RuntimeError("Media to Pose test suite failed")


if __name__ == "__main__":
    main()
