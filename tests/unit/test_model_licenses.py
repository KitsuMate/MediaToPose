"""Licence acceptance stored per model revision in the storage marker."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from media_to_pose.backends import licenses, lifecycle
from media_to_pose.core.errors import InvalidRequest, LicenseRequired

LICENSE = licenses.ModelLicense(
    "test-license", "Test License", "Use it kindly.", "https://example.invalid/l"
)


class LicenseTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.storage = lifecycle.ManagedStorage(
            "media_to_pose_test", "test_models", "Test"
        )
        self.root = self.storage.root(Path(directory.name))

    def test_unaccepted_licence_names_the_exact_accept_value(self) -> None:
        with self.assertRaises(LicenseRequired) as caught:
            licenses.ensure_accepted(self.storage, self.root, LICENSE, "a" * 40)
        details = caught.exception.details
        self.assertEqual(details["accept_license"], f"test-license@{'a' * 40}")
        self.assertEqual(details["url"], "https://example.invalid/l")
        self.assertEqual(details["summary"], "Use it kindly.")
        self.assertEqual(caught.exception.exit_code, 10)

    def test_acceptance_is_recorded_per_revision_with_source(self) -> None:
        revision = "a" * 40
        entry = licenses.ensure_accepted(
            self.storage,
            self.root,
            LICENSE,
            revision,
            accept=LICENSE.accept_value(revision),
            source=licenses.AcceptanceSource.CLI,
        )
        self.assertEqual(entry["source"], "CLI")
        self.assertTrue(entry["accepted_at"].endswith("Z"))
        again = licenses.ensure_accepted(self.storage, self.root, LICENSE, revision)
        self.assertEqual(again, entry)
        with self.assertRaises(LicenseRequired):
            licenses.ensure_accepted(self.storage, self.root, LICENSE, "b" * 40)

    def test_acceptance_survives_re_marking_the_storage(self) -> None:
        revision = "c" * 40
        licenses.record_acceptance(
            self.storage, self.root, LICENSE, revision, licenses.AcceptanceSource.PANEL
        )
        self.storage.ensure(self.root)
        marker = json.loads((self.root / lifecycle.STORAGE_MARKER).read_text())
        self.assertEqual(marker["owner"], "media_to_pose")
        self.assertIn(LICENSE.accept_value(revision), marker["licenses"])
        self.assertIsNotNone(
            licenses.acceptance(self.storage, self.root, LICENSE, revision)
        )

    def test_mismatched_accept_value_is_refused(self) -> None:
        with self.assertRaises(InvalidRequest):
            licenses.ensure_accepted(
                self.storage, self.root, LICENSE, "a" * 40, accept="test-license@main"
            )
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
