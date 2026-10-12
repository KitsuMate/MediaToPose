"""Real MediaPipe face detection on the face stills, checked against what each
still visibly shows (tests/fixtures/expected/face_expressions.json)."""

from __future__ import annotations

import json
import operator
import unittest

from tests.blender_test_support import (
    FIXTURES,
    fixture_manifest,
    load_addon,
    verify_fixture,
)

OPERATORS = {">=": operator.ge, "<=": operator.le}


def _value(blendshapes: dict[str, float], expression: str) -> float:
    if " - " in expression:
        first, second = expression.split(" - ")
        return blendshapes.get(first, 0.0) - blendshapes.get(second, 0.0)
    return blendshapes.get(expression, 0.0)


class FaceDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()
        expected_path = verify_fixture("face_expressions_expected")
        cls.expected = json.loads(expected_path.read_text(encoding="utf8"))["stills"]
        provider_module = cls.addon.providers.mediapipe.provider
        cls.provider = provider_module.MediaPipeProvider(
            str(cls.addon.core.paths.BUNDLED_MODELS),
            use_body=True,
            use_hands=False,
            use_face=True,
            smoothing=0.0,
        )
        cls.detected = {}
        for entry in fixture_manifest()["fixture"]:
            if entry["id"] in cls.expected:
                path = verify_fixture(entry["id"])
                cls.detected[entry["id"]] = cls.provider.detect_image(str(path))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.provider.close()

    def test_every_still_has_an_expectation_and_a_licence(self) -> None:
        stills = {
            entry["id"]: entry
            for entry in fixture_manifest()["fixture"]
            if entry["path"].startswith("media/face/")
        }
        self.assertEqual(set(stills), set(self.expected))
        for entry in stills.values():
            with self.subTest(still=entry["id"]):
                self.assertEqual(entry["license"], "CC0-1.0")
                self.assertTrue(entry["source_url"].startswith("https://"))
                self.assertTrue((FIXTURES / entry["path"]).is_file())

    def test_mirrored_input_mirrors_the_face_like_the_body(self) -> None:
        provider_module = self.addon.providers.mediapipe.provider
        provider = provider_module.MediaPipeProvider(
            str(self.addon.core.paths.BUNDLED_MODELS),
            use_body=True,
            use_hands=False,
            use_face=True,
            mirror=True,
            smoothing=0.0,
        )
        try:
            mirrored = provider.detect_image(str(verify_fixture("face_wink_left")))
        finally:
            provider.close()
        plain = self.detected["face_wink_left"].blendshapes
        flipped = mirrored.blendshapes
        # The left wink becomes a right wink, as the body's left becomes the
        # character's right.
        self.assertGreater(plain["eyeSquintLeft"], plain["eyeSquintRight"])
        self.assertGreater(flipped["eyeSquintRight"], flipped["eyeSquintLeft"])

    def test_detected_expressions_match_the_stills(self) -> None:
        unexpected_passes = []
        for still_id, expectation in self.expected.items():
            frame = self.detected[still_id]
            with self.subTest(still=still_id):
                self.assertTrue(frame.blendshapes, f"{still_id}: no face detected")
                failures = [
                    f"{expression} {sign} {limit} (got "
                    f"{_value(frame.blendshapes, expression):.2f})"
                    for expression, sign, limit in expectation["checks"]
                    if not OPERATORS[sign](_value(frame.blendshapes, expression), limit)
                ]
                if expectation.get("known_issue"):
                    if not failures:
                        unexpected_passes.append(still_id)
                    continue
                self.assertEqual(failures, [], f"{still_id}: {expectation['visible']}")
        self.assertEqual(
            unexpected_passes,
            [],
            "These stills now pass; remove their known_issue entry",
        )


if __name__ == "__main__":
    unittest.main()
