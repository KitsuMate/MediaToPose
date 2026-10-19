"""Free disk and memory checks shared by the model backends."""

from __future__ import annotations

import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from media_to_pose.backends import resources
from media_to_pose.core.errors import InsufficientResources

_Usage = namedtuple("_Usage", "total used free")


class DiskTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_missing_folders_resolve_to_their_existing_parent(self) -> None:
        missing = self.root / "not" / "yet" / "there"
        self.assertEqual(
            resources.closest_existing_directory(missing), self.root.resolve()
        )
        self.assertTrue(resources.same_volume(missing, self.root))
        self.assertGreater(resources.free_disk_bytes(missing), 0)

    def test_shortfall_reports_needed_and_available_bytes(self) -> None:
        with mock.patch.object(
            resources.shutil, "disk_usage", return_value=_Usage(10, 9, 1000)
        ):
            resources.require_disk(self.root, 1000, "Model")
            with self.assertRaises(InsufficientResources) as caught:
                resources.require_disk(
                    self.root / "model", 5000, "Model", fitting_variant="INT8"
                )
        details = caught.exception.details
        self.assertEqual(details["resource"], "disk")
        self.assertEqual(details["needed_bytes"], 5000)
        self.assertEqual(details["available_bytes"], 1000)
        self.assertEqual(details["fitting_variant"], "INT8")
        self.assertEqual(caught.exception.exit_code, 11)


if __name__ == "__main__":
    unittest.main()
