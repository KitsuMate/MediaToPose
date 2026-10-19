"""Free disk and memory checks shared by the model backends."""

from __future__ import annotations

import sys
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


class MemoryTests(unittest.TestCase):
    def test_this_machine_reports_available_memory(self) -> None:
        if sys.platform not in {"win32", "darwin"} and not sys.platform.startswith(
            "linux"
        ):
            self.skipTest("unsupported platform")
        available = resources.available_ram_bytes()
        self.assertIsNotNone(available)
        self.assertGreater(available or 0, 64 * 1024**2)

    def test_linux_reads_mem_available(self) -> None:
        meminfo = "MemTotal: 32000000 kB\nMemFree: 1000 kB\nMemAvailable: 8000 kB\n"
        self.assertEqual(resources._linux_available_ram(meminfo), 8000 * 1024)
        with self.assertRaises(OSError):
            resources._linux_available_ram("MemTotal: 1 kB\n")

    def test_macos_scales_memory_by_the_pressure_level(self) -> None:
        values = {"hw.memsize": 16 * 1024**3, "kern.memorystatus_level": 50}
        with mock.patch.object(resources, "_sysctl_int", side_effect=values.get):
            self.assertEqual(resources.available_ram_bytes("darwin"), 8 * 1024**3)

    def test_unknown_memory_passes_and_a_shortfall_raises(self) -> None:
        with mock.patch.object(resources, "available_ram_bytes", return_value=None):
            resources.require_ram(10**15, "Encoder")
        with (
            mock.patch.object(resources, "available_ram_bytes", return_value=4 * 10**9),
            self.assertRaises(InsufficientResources) as caught,
        ):
            resources.require_ram(8 * 10**9, "Encoder")
        self.assertEqual(caught.exception.details["resource"], "ram")
        self.assertEqual(caught.exception.details["needed_bytes"], 8 * 10**9)
        self.assertEqual(caught.exception.details["available_bytes"], 4 * 10**9)

    def test_failed_query_is_unknown(self) -> None:
        with mock.patch.object(
            resources.MEMINFO.__class__, "read_text", side_effect=OSError("denied")
        ):
            self.assertIsNone(resources.available_ram_bytes("linux"))
        self.assertIsNone(resources.available_ram_bytes("sunos5"))


if __name__ == "__main__":
    unittest.main()
