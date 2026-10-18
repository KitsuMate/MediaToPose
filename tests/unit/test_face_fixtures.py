"""Every face still has a licence, a source and an expectation."""

from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FaceFixtureTests(unittest.TestCase):
    def test_every_still_has_an_expectation_and_a_licence(self) -> None:
        with (FIXTURES / "fixtures.toml").open("rb") as stream:
            manifest = tomllib.load(stream)
        expected = json.loads(
            (FIXTURES / "expected" / "face_expressions.json").read_text("utf-8")
        )["stills"]
        stills = {
            entry["id"]: entry
            for entry in manifest["fixture"]
            if entry["path"].startswith("media/face/")
        }
        self.assertEqual(set(stills), set(expected))
        for entry in stills.values():
            with self.subTest(still=entry["id"]):
                self.assertEqual(entry["license"], "CC0-1.0")
                self.assertTrue(entry["source_url"].startswith("https://"))
                self.assertEqual(len(entry["crop"]), 4)
                self.assertTrue((FIXTURES / entry["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
