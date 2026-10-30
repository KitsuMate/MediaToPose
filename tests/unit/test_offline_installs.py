"""Model installs honour Blender's online access setting before any work."""

from __future__ import annotations

import unittest
from unittest import mock

from media_to_pose.backends.anycalib import backend as anycalib_backend
from media_to_pose.backends.rohm import backend as rohm_backend
from media_to_pose.backends.sam3d import backend as sam3d_backend
from media_to_pose.core.errors import OnlineAccessDisabled


class OfflineInstallTests(unittest.TestCase):
    def test_installs_are_refused_while_offline(self) -> None:
        installs = {
            "AnyCalib": lambda: anycalib_backend.start_install(online_access=False),
            "RoHM": lambda: rohm_backend.start_install(online_access=False),
            "SAM 3D Body": lambda: sam3d_backend.start_install(
                "VITH", "AUTO", online_access=False
            ),
        }
        for label, install in installs.items():
            with (
                self.subTest(model=label),
                mock.patch.object(
                    sam3d_backend.BACKEND, "start_job", side_effect=AssertionError
                ),
                mock.patch.object(
                    anycalib_backend.BACKEND, "start_job", side_effect=AssertionError
                ),
                mock.patch.object(
                    rohm_backend.BACKEND, "start_job", side_effect=AssertionError
                ),
                self.assertRaisesRegex(OnlineAccessDisabled, label),
            ):
                install()


if __name__ == "__main__":
    unittest.main()
