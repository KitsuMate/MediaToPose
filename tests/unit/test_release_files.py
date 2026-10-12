"""The release package carries copies of the repository's license files."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPIED = ("LICENSE", "README.md", "THIRD_PARTY_NOTICES.md")


class ReleaseFileTests(unittest.TestCase):
    def test_packaged_copies_match_the_repository(self) -> None:
        for name in COPIED:
            with self.subTest(name=name):
                original = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                packaged = (ROOT / "media_to_pose" / name).read_bytes()
                self.assertEqual(
                    packaged.replace(b"\r\n", b"\n"),
                    original,
                    f"Copy {name} into media_to_pose/ after editing it",
                )


if __name__ == "__main__":
    unittest.main()
